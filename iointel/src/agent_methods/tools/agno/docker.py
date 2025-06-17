from typing import Dict, Optional, Union

from pydantic import Field
from agno.tools.docker import DockerTools as AgnoDockerTools

from .common import make_base, wrap_tool


class Docker(make_base(AgnoDockerTools)):
    enable_container_management: bool = Field(required=True, frozen=True)
    enable_image_management: bool = Field(required=True, frozen=True)
    enable_volume_management: bool = Field(required=False, frozen=True)
    enable_network_management: bool = Field(required=False, frozen=True)

    def _get_tool(self):
        return self.Inner(
            enable_container_management=self.enable_container_management,
            enable_image_management=self.enable_image_management,
            enable_volume_management=self.enable_volume_management,
            enable_network_management=self.enable_network_management,
        )

    @wrap_tool("agno__docker__list_containers", AgnoDockerTools.list_containers)
    def list_containers(self, all: bool = False) -> str:
        return self._tool.list_containers(all)

    @wrap_tool("agno__docker__start_container", AgnoDockerTools.start_container)
    def start_container(self, container_id: str) -> str:
        return self._tool.start_container(container_id)

    @wrap_tool("agno__docker__stop_container", AgnoDockerTools.stop_container)
    def stop_container(self, container_id: str, timeout: int = 10) -> str:
        return self._tool.stop_container(container_id, timeout)

    @wrap_tool("agno__docker__remove_container", AgnoDockerTools.remove_container)
    def remove_container(
        self, container_id: str, force: bool = False, volumes: bool = False
    ) -> str:
        return self._tool.remove_container(container_id, force, volumes)

    @wrap_tool("agno__docker__get_container_logs", AgnoDockerTools.get_container_logs)
    def get_container_logs(
        self, container_id: str, tail: int = 100, stream: bool = False
    ) -> str:
        return self._tool.get_container_logs(container_id, tail, stream)

    @wrap_tool("agno__docker__inspect_container", AgnoDockerTools.inspect_container)
    def inspect_container(self, container_id: str) -> str:
        return self._tool.inspect_container(container_id)

    @wrap_tool("agno__docker__run_container", AgnoDockerTools.run_container)
    def run_container(
        self,
        image: str,
        command: Optional[str] = None,
        name: Optional[str] = None,
        detach: bool = True,
        ports: Optional[Dict[str, Union[str, int]]] = None,  # Updated type hint
        volumes: Optional[Dict[str, Dict[str, str]]] = None,
        environment: Optional[Dict[str, str]] = None,
        network: Optional[str] = None,
    ) -> str:
        return self._tool.run_container(
            image, command, name, detach, ports, volumes, environment, network
        )

    @wrap_tool("agno__docker__exec_in_container", AgnoDockerTools.exec_in_container)
    def exec_in_container(self, container_id: str, command: str) -> str:
        return self._tool.exec_in_container(container_id, command)

    @wrap_tool("agno__docker__list_images", AgnoDockerTools.list_images)
    def list_images(self) -> str:
        return self._tool.list_images()

    @wrap_tool("agno__docker__pull_image", AgnoDockerTools.pull_image)
    def pull_image(self, image_name: str, tag: str = "latest") -> str:
        return self._tool.pull_image(image_name, tag)

    @wrap_tool("agno__docker__remove_image", AgnoDockerTools.remove_image)
    def remove_image(self, image_id: str, force: bool = False) -> str:
        return self._tool.remove_image(image_id, force)

    @wrap_tool("agno__docker__build_image", AgnoDockerTools.build_image)
    def build_image(
        self, path: str, tag: str, dockerfile: str = "Dockerfile", rm: bool = True
    ) -> str:
        return self._tool.build_image(path, tag, dockerfile, rm)

    @wrap_tool("agno__docker__tag_image", AgnoDockerTools.tag_image)
    def tag_image(
        self, image_id: str, repository: str, tag: Optional[str] = None
    ) -> str:
        return self._tool.tag_image(image_id, repository, tag)

    @wrap_tool("agno__docker__inspect_image", AgnoDockerTools.inspect_image)
    def inspect_image(self, image_id: str) -> str:
        return self._tool.inspect_image(image_id)

    @wrap_tool("agno__docker__list_volumes", AgnoDockerTools.list_volumes)
    def list_volumes(self) -> str:
        return self._tool.list_volumes()

    @wrap_tool("agno__docker__create_volume", AgnoDockerTools.create_volume)
    def create_volume(
        self,
        volume_name: str,
        driver: str = "local",
        labels: Optional[Dict[str, str]] = None,
    ) -> str:
        return self._tool.create_volume(volume_name, driver, labels)

    @wrap_tool("agno__docker__remove_volume", AgnoDockerTools.remove_volume)
    def remove_volume(self, volume_name: str, force: bool = False) -> str:
        return self._tool.remove_volume(volume_name, force)

    @wrap_tool("agno__docker__inspect_volume", AgnoDockerTools.inspect_volume)
    def inspect_volume(self, volume_name: str) -> str:
        return self._tool.inspect_volume(volume_name)

    @wrap_tool("agno__docker__list_networks", AgnoDockerTools.inspect_volume)
    def list_networks(self) -> str:
        return self._tool.list_networks()

    @wrap_tool("agno__docker__create_network", AgnoDockerTools.create_network)
    def create_network(
        self,
        network_name: str,
        driver: str = "bridge",
        internal: bool = False,
        labels: Optional[Dict[str, str]] = None,
    ) -> str:
        return self.create_network(network_name, driver, internal, labels)

    @wrap_tool("agno__docker__remove_network", AgnoDockerTools.create_network)
    def remove_network(self, network_name: str) -> str:
        return self.remove_network(network_name)

    @wrap_tool("agno__docker__inspect_network", AgnoDockerTools.inspect_network)
    def inspect_network(self, network_name: str) -> str:
        return self.inspect_network(network_name)

    @wrap_tool(
        "agno__docker__connect_container_to_network",
        AgnoDockerTools.connect_container_to_network,
    )
    def connect_container_to_network(self, container_id: str, network_name: str) -> str:
        return self._tool.connect_container_to_network(container_id, network_name)

    @wrap_tool(
        "agno__docker__disconnect_container_from_network",
        AgnoDockerTools.disconnect_container_from_network,
    )
    def disconnect_container_from_network(
        self, container_id: str, network_name: str
    ) -> str:
        return self._tool.disconnect_container_from_network(container_id, network_name)
