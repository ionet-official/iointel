import tempfile
import os
import docker
from typing import Tuple


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

    def run_code_in_sandbox(self, code: str) -> Tuple[str, str, int]:
        """
        Run the given code in a sandboxed Docker container with gVisor runtime and enhanced security.

        Security Measures:
        - gVisor runtime (runsc): Provides additional isolation layer.
        - no-new-privileges: Prevents privilege escalation.
        - Non-root user: Running as 'nobody'.
        - Dropped capabilities: Removes all Linux capabilities.
        - Read-only filesystem: Container's filesystem is read-only.
        - Network disabled: No outbound network (optional, controlled by network_disabled flag).
        - CPU and memory limits to prevent resource exhaustion.

        Returns: (stdout, stderr, exit_code)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "script.py")
            with open(script_path, "w") as f:
                f.write(code)

            # Ensure the Docker image is available (pull if needed)
            self.client.images.pull(self.image)

            container = self.client.containers.run(
                self.image,
                command=["python", "script.py"],
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
                runtime=self.runtime,  # Utilize gVisor runtime
                # If you need a writable temp directory, you can add:
                # tmpfs={"/tmp": "size=64m"}
            )

            exit_code = container.wait()
            logs = container.logs()
            container.remove(force=True)

            stdout = logs.decode("utf-8")
            stderr = ""  # Without splitting stdout/err, we consider all logs stdout
            if isinstance(exit_code, dict):
                exit_code = exit_code.get('StatusCode', 1)

            return stdout, stderr, exit_code
