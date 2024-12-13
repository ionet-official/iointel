import tempfile
import os
import docker
from typing import Tuple, List
from code_parser import (PythonModule, extract_imported_modules,
                         generate_code_from_module, filter_packages)

class DockerSandbox:
    def __init__(
        self, 
        image: str = "python:3.11-slim", 
        memory_limit="100m", 
        cpu_period=100000, 
        cpu_quota=50000,
        read_only=True,
        network_disabled=True,
        cap_drop=["ALL"],
        user="nobody",
        security_opt=["no-new-privileges"],
        runtime="runsc"  # Use gVisor runtime
    ):
        """
        Initialize a DockerSandbox environment with gVisor and additional security measures.

        :param image: Docker image to use for running the code.
        :param memory_limit: Memory limit for the container, e.g., '100m' for 100 MB.
        :param cpu_period: CPU period for quota control.
        :param cpu_quota: CPU quota for limiting CPU usage.
        :param read_only: If True, mount container's filesystem as read-only.
        :param network_disabled: If True, container will not have network access.
        :param cap_drop: List of capabilities to drop. Default to all.
        :param user: User to run as inside container, 'nobody' is a non-privileged user.
        :param security_opt: Security options, 'no-new-privileges' prevents privilege escalation.
        :param runtime: Docker runtime to use, 'runsc' to leverage gVisor for isolation.
        """
        self.image = image
        self.memory_limit = memory_limit
        self.cpu_period = cpu_period
        self.cpu_quota = cpu_quota
        self.read_only = read_only
        self.network_disabled = network_disabled
        self.cap_drop = cap_drop
        self.user = user
        self.security_opt = security_opt
        self.runtime = runtime
        self.client = docker.from_env()


    def run_code_in_sandbox(self, module: PythonModule) -> Tuple[str, str, int]:
        """
        Run the given PythonModule code object in a sandboxed Docker container.

        This function:
        1. Converts the PythonModule to a Python code string.
        2. Extracts imported modules, determines which ones need installation.
        3. Creates a temporary directory on the host.
        4. Writes the generated code into `script.py`.
        5. Starts a Docker container using configured image and security settings.
        6. Installs non-standard packages via `pip install` before running the script, if needed.
        7. Executes `script.py` inside the container.
        8. Captures and returns stdout, stderr, and exit code.

        Parameters
        ----------
        module : PythonModule
            The Pydantic model representing the Python module's structure.

        Returns
        -------
        Tuple[str, str, int]
            A tuple containing:
            - stdout (str): The standard output from execution.
            - stderr (str): The standard error output (currently empty, as not separately captured).
            - exit_code (int): The exit code from the container process. 0 indicates success.

        Notes
        -----
        - Security measures applied include no-new-privileges, non-root user, dropped capabilities,
          optional read-only filesystem, and limited resources.
        - If `network_disabled` is True, the code will not have internet access.
        - Packages are installed with pip if they are not recognized as standard libraries.
        - The container is removed after execution, leaving no persistent state.

        Raises
        ------
        docker.errors.APIError
            If there is an error pulling the image, creating, or running the container.
        """

        # Generate code from the PythonModule
        code = generate_code_from_module(module)
        # Extract imported modules
        modules_to_install = extract_imported_modules(module)
        packages = filter_packages(modules_to_install)

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "script.py")
            with open(script_path, "w") as f:
                f.write(code)

            self.client.images.pull(self.image)

            # Construct the command to run inside the container
            cmd_parts = []
            if packages:
                cmd_parts.append("pip install " + " ".join(packages))
            cmd_parts.append("python script.py")
            full_cmd = " && ".join(cmd_parts)

            container = self.client.containers.run(
                self.image,
                command=["/bin/sh", "-c", full_cmd],
                working_dir="/app",
                volumes={tmpdir: {'bind': '/app', 'mode': 'ro'}},
                stdin_open=False,
                tty=False,
                detach=True,
                mem_limit=self.memory_limit,
                cpu_period=self.cpu_period,
                cpu_quota=self.cpu_quota,
                security_opt=self.security_opt,
                user=self.user,
                network_disabled=self.network_disabled,
                read_only=self.read_only,
                cap_drop=self.cap_drop,
                runtime=self.runtime,
            )

            exit_code = container.wait()
            logs = container.logs()
            container.remove(force=True)

            stdout = logs.decode("utf-8")
            stderr = ""
            if isinstance(exit_code, dict):
                exit_code = exit_code.get('StatusCode', 1)

            return stdout, stderr, exit_code