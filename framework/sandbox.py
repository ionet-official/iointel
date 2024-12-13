import tempfile
import os
import docker
from typing import Tuple, List
import logging
from code_parser import (PythonModule, extract_imported_modules,
                         generate_code_from_module, filter_packages)

# Configure logging for this module. In a larger application, configure logging in a main entry point.
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set this to INFO or WARNING in production if needed.
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

class DockerSandbox:
    def __init__(
        self, 
        image: str = "python:3.11-slim", 
        memory_limit="500m", 
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
        logger.debug("Generating code from PythonModule...")
        code = generate_code_from_module(module)
        logger.debug("Code generation complete, length=%d characters", len(code))

        # Extract imported modules
        logger.debug("Extracting imported modules...")
        modules_to_install = extract_imported_modules(module)
        logger.debug("Imported modules: %s", modules_to_install)

        # Determine packages to install
        packages = filter_packages(modules_to_install)
        logger.debug("Packages to install (after filtering stdlib): %s", packages)

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "script.py")
            with open(script_path, "w") as f:
                f.write(code)
            os.chmod(script_path, 0o644)
            os.chmod(tmpdir, 0o755)
            logger.info("Wrote generated code to temporary file: %s", script_path)

            logger.debug("Pulling Docker image %s...", self.image)
            try:
                self.client.images.pull(self.image)
                logger.info("Image %s pulled successfully", self.image)
            except docker.errors.APIError as e:
                logger.error("Failed to pull image %s: %s", self.image, str(e))
                raise

            # Construct the command to run inside the container
            cmd_parts = []
            if packages:
                cmd_parts.append("pip install --no-cache-dir --user " + " ".join(packages))
            cmd_parts.append("python script.py")
            full_cmd = " && ".join(cmd_parts)
            logger.debug("Full command to run in container: %s", full_cmd)

            logger.debug("Creating and starting container...")
            try:
                container = self.client.containers.run(
                    self.image,
                    command=["/bin/sh", "-c", full_cmd],
                    working_dir="/app",
                    volumes={tmpdir: {'bind': '/app', 'mode': 'ro'}},
                    tmpfs={"/home/nobody": "size=128m"},  # Provide a writable home directory
                    environment={"HOME": "/home/nobody"},  # Set HOME to the writable directory
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
                logger.info("Container started successfully. ID: %s", container.id)
            except docker.errors.APIError as e:
                logger.error("Failed to start container: %s", str(e))
                raise

            logger.debug("Waiting for container to finish execution...")
            exit_code = container.wait()
            logger.debug("Container finished with raw exit code data: %s", exit_code)

            logs = container.logs()
            logger.debug("Collected logs from container (length=%d)", len(logs))

            # Clean up container
            container.remove(force=True)
            logger.info("Container removed.")

            stdout = logs.decode("utf-8")
            stderr = ""
            if isinstance(exit_code, dict):
                exit_code = exit_code.get('StatusCode', 1)

            logger.info("Execution completed. Exit code: %d", exit_code)
            logger.debug("STDOUT: %s", stdout)

            return stdout, stderr, exit_code