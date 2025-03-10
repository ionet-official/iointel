import tempfile
import os
import docker
from typing import Tuple, Union
import logging
from iointel.src.code_parsers.pycode_parser import PythonModule, PythonCodeGenerator
from iointel.src.code_parsers.jscode_parser import (
    JavaScriptModule,
    JavaScriptCodeGenerator,
)
from pydantic import ValidationError

# Configure logging for this module. In a larger application, configure logging in a main entry point.
logger = logging.getLogger(__name__)
# Fallback to "DEBUG" if not set
level_name = os.environ.get("LOGGING_LEVEL", "DEBUG")
level_name = level_name.upper()
# Safely get the numeric logging level, default to DEBUG if invalid
numeric_level = getattr(logging, level_name, logging.DEBUG)
logger.setLevel(numeric_level)
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
        runtime="runsc",  # Use gVisor runtime
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

    def validate_module(self, module: str) -> Union[PythonModule, JavaScriptModule]:
        """
        Validate the input module string as Python or JavaScript code.

        :param module: The code string to validate.
        :return: A Pydantic model instance representing the module.
        :raises ValidationError: If the module is invalid or unsupported.
        """
        try:
            logger.debug("Validating module as PythonModule...")
            py_module = PythonModule.model_validate_json(module)
            logger.info("Module validated as PythonModule.")
            return py_module
        except ValidationError as e:
            logger.debug("Validation as PythonModule failed: %s", str(e))
            try:
                logger.debug("Validating module as JavaScriptModule...")
                js_module = JavaScriptModule.model_validate_json(module)
                logger.info("Module validated as JavaScriptModule.")
                return js_module
            except ValidationError as e:
                logger.error(
                    "Validation failed for both PythonModule and JavaScriptModule."
                )

    def run_code_in_sandbox(self, module_json: str) -> Tuple[str, str, int]:
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
        module : str
            The json dump of the Pydantic model representing the Python module's structure.

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
        module = self.validate_module(module_json)

        if isinstance(module, PythonModule):
            logger.debug("Module detected as Python.")
            # Generate Python code
            py_gen = PythonCodeGenerator()
            logger.debug("Generating Python code from PythonModule...")
            code = py_gen.generate_pycode_from_module(module)
            logger.debug("Code generation complete, length=%d characters", len(code))

            # Extract imported modules and install packages if needed
            logger.debug("Extracting imported modules...")
            modules_to_install = py_gen.extract_imported_modules(module)
            logger.debug("Imported modules: %s", modules_to_install)

            packages = py_gen.filter_packages(modules_to_install)
            logger.debug("Packages to install (for Python): %s", packages)

            runtime_cmd = "python script.py"
            install_cmd = (
                f"pip install --no-cache-dir --user {' '.join(packages)}"
                if packages
                else ""
            )
            container_image = self.image
            script_name = "script.py"
            extra_env = {"HOME": "/home/nobody"}
            extra_tmpfs = {"/home/nobody": "size=128m"}

        elif isinstance(module, JavaScriptModule):
            logger.debug("Module detected as JavaScript.")
            logger.debug("Generating JavaScript code from JavaScriptModule...")
            js_gen = JavaScriptCodeGenerator()
            code = js_gen.generate_code_from_js_module(module)
            logger.debug("Code generation complete, length=%d characters", len(code))

            # Extract imported modules and install packages if needed
            logger.debug("Extracting imported modules...")
            js_packages = js_gen.extract_imported_modules(module)
            logger.debug("Imported modules: %s", js_packages)

            logger.debug("Packages to install (for JavaScript): %s", js_packages)

            runtime_cmd = "node script.js"
            # Install JS packages into /app/vendor
            # Use npm with --prefix to install packages into /app/vendor
            install_cmd = (
                f"npm install --prefix /app/vendor {' '.join(js_packages)}"
                if js_packages
                else ""
            )

            # Use Node.js image
            container_image = "node:18-slim"
            script_name = "script.js"

            # For JS, we provide writable vendor directory and set NODE_PATH
            extra_env = {
                "HOME": "/home/nobody",
                "NODE_PATH": "/app/vendor/node_modules",
            }
            # tmpfs for writable vendor directory and home
            extra_tmpfs = {"/home/nobody": "size=128m", "/app/vendor": "size=128m"}

        else:
            raise ValueError(
                "Unsupported module type. Must be PythonModule or JavaScriptModule."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, script_name)
            with open(script_path, "w") as f:
                f.write(code)

            # Ensure world-readable permissions
            os.chmod(script_path, 0o644)
            os.chmod(tmpdir, 0o755)
            logger.info("Wrote generated code to temporary file: %s", script_path)

            logger.debug("Pulling Docker image %s...", container_image)
            try:
                self.client.images.pull(container_image)
                logger.info("Image %s pulled successfully", container_image)
            except docker.errors.APIError as e:
                logger.error("Failed to pull image %s: %s", container_image, str(e))
                raise

            # Construct the full command
            cmd_parts = []
            if install_cmd:
                cmd_parts.append(install_cmd)
            cmd_parts.append(runtime_cmd)
            full_cmd = " && ".join(cmd_parts) if cmd_parts else runtime_cmd
            logger.debug("Full command to run in container: %s", full_cmd)

            logger.debug("Creating and starting container...")
            try:
                container = self.client.containers.run(
                    container_image,
                    command=["/bin/sh", "-c", full_cmd],
                    working_dir="/app",
                    volumes={tmpdir: {"bind": "/app", "mode": "ro"}},
                    tmpfs=extra_tmpfs,
                    environment=extra_env,
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
                exit_code = exit_code.get("StatusCode", 1)

            logger.info("Execution completed. Exit code: %d", exit_code)
            logger.debug("STDOUT: %s", stdout)

            return stdout, stderr, exit_code
