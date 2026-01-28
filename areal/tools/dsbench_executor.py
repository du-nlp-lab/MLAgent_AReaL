import os
import subprocess
import tempfile


class DSBenchExecutor:
    def execute(self, code, work_dir):
        with tempfile.NamedTemporaryFile(
            suffix=".py", dir=work_dir, mode="w", delete=False
        ) as f:
            f.write(code)
            script_path = f.name

        try:
            result = subprocess.run(
                ["python3", script_path],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                obs = f"Success.\nOutput: {result.stdout[:1000]}"  # 截断保护
            else:
                obs = f"Failed.\nError: {result.stderr}"
        except Exception as e:
            obs = f"Runtime Error: {str(e)}"
        finally:
            if os.path.exists(script_path):
                os.remove(script_path)

        return {"observation": obs}
