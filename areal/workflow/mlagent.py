from areal.utils import logging

logger = logging.getLogger("MLRCWorkflow")


class MLRCWorkflow:
    def __init__(self, executor, judge_client):
        """
        Workflow manager for MLRC tasks.
        executor: MLRCExecutor instance.
        judge_client: GPT-4o client for auditing and error fixing.
        """
        self.executor = executor
        self.judge_client = judge_client

    def _fix_dependency_error(self, stderr):
        """
        Internal: Uses GPT-4o to analyze the error and return a fix command.
        """
        logger.info("Analyzing error with GPT-4o for self-healing...")

        prompt = f"""
        As an ML Infrastructure expert, analyze this execution error:
        ---
        {stderr[-2000:]}
        ---
        Based on the error (e.g., ModuleNotFoundError, Version Conflict),
        provide the EXACT shell command to fix it (e.g., 'pip install pandas').
        Output ONLY the shell command.
        """

        try:
            # Reusing the judge_client (GPT-4o) to generate the fix
            fix_command = self.judge_client.generate(prompt).strip()
            # Safety check: only allow pip/conda/apt commands
            if any(
                cmd in fix_command.lower()
                for cmd in ["pip", "conda", "apt", "python -m"]
            ):
                return fix_command
            return None
        except Exception as e:
            logger.error(f"Failed to generate fix: {e}")
            return None

    def execute_research_step(self, work_dir, script_name="reproduce_paper.py"):
        """
        The core loop of running a research attempt.
        1. Run the script.
        2. If it fails due to environment issues, try to fix it once.
        3. Return final result.
        """
        # Step 1: Initial Execution
        command = f"python {script_name}"
        result = self.executor.run_research_command(work_dir, command)

        # Step 2: Error Detection & Self-Healing
        # Check for common dependency/environment markers in stderr
        error_markers = [
            "ModuleNotFoundError",
            "ImportError",
            "pkg_resources.DistributionNotFound",
        ]

        if any(marker in result["stderr"] for marker in error_markers):
            logger.warning("Dependency error detected. Attempting self-healing...")

            fix_cmd = self._fix_dependency_error(result["stderr"])
            if fix_cmd:
                logger.info(f"Applying fix: {fix_cmd}")
                # Execute the fix (e.g., pip install)
                self.executor.run_research_command(work_dir, fix_cmd)

                # Step 3: Retry the original command after fixing
                logger.info("Retrying original research script...")
                result = self.executor.run_research_command(work_dir, command)

        return result

    def run_rollout(self, task_data):
        """
        High-level entry for AReaL rollout workers.
        """
        source_env = task_data["work_dir"]
        # Create isolated sandbox
        sandbox_dir = self.executor.setup_workspace(source_env)

        try:
            # Here we assume the Agent's generated code is saved as 'reproduce_paper.py'
            # In a real GRPO loop, the code comes from the model's policy sampling.
            result = self.execute_research_step(sandbox_dir)
            return result
        finally:
            # Ensure the temporary workspace is deleted to save disk
            self.executor.cleanup(sandbox_dir)
