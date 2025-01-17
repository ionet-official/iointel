from controlflow.orchestration.handler import AsyncHandler,Handler
from controlflow.events.events import (
    UserMessage,
    AgentMessage,
    OrchestratorMessage,
    ToolCall,
    ToolResult,
    OrchestratorStart,
    OrchestratorEnd,
    OrchestratorError,
    EndTurn,
    AgentMessageDelta,
    Event
)

from framework.src.agent_methods.data_models.datamodels import ( 
    AgentMessageEvent,
    UserMessageEvent,
    OrchestratorMessageEvent, 
    ToolCallEvent,
    ToolResultEvent,
    OrchestratorStartEvent, 
    OrchestratorEndEvent,
    AgentMessageDeltaEvent,
    OrchestratorErrorEvent, 
    EndTurnEvent,
    EventsLog 
    )



class LoggingHandler(Handler):
    def __init__(self):
        self.log = EventsLog()

    def on_agent_message(self, event: AgentMessage):
        self.log.events.append(
            AgentMessageEvent(
                agent_name=event.agent.name,
                content=event.ai_message.content
            )
        )

    def on_user_message(self, event: UserMessage):
        self.log.events.append(
            UserMessageEvent(
                content=event.content
            )
        )

    def on_orchestrator_message(self, event: OrchestratorMessage):
        self.log.events.append(
            OrchestratorMessageEvent(
                content=event.content
            )
        )

    def on_tool_call(self, event: ToolCall):
        self.log.events.append(
            ToolCallEvent(
                tool_name=event.tool_name
            )
        )

    def on_tool_result(self, event: ToolResult):
        self.log.events.append(
            ToolResultEvent(
                tool_name=event.tool_name,
                result=event.result
            )
        )

    def on_orchestrator_start(self, event: OrchestratorStart):
        self.log.events.append(
            OrchestratorStartEvent()
        )

    def on_orchestrator_end(self, event: OrchestratorEnd):
        self.log.events.append(
            OrchestratorEndEvent()
        )

    def on_agent_message_delta(self, event: AgentMessageDelta):
        self.log.events.append(
            AgentMessageDeltaEvent(
                delta=event.delta
            )
        )

    def on_orchestrator_error(self, event: OrchestratorError):
        self.log.events.append(
            OrchestratorErrorEvent(
                error=str(event.error)
            )
        )

    def on_end_turn(self, event: EndTurn):
        self.log.events.append(
            EndTurnEvent()
        )

    def on_event(self, event: Event):
        """
        Fallback for unhandled events if you want to store them too,
        or do something generic. The default Handler base class calls
        this method for all events. We can ignore or store it.
        """
        # self.log.events.append( ...some default model... )
        pass

    def get_log(self) -> EventsLog:
        """
        Expose the Pydantic log container for consumption.
        """
        return self.log


class AsyncLoggingHandler(AsyncHandler):

    def __init__(self):
        self.log = EventsLog()

    async def on_agent_message(self, event: AgentMessage):
        self.log.events.append(
            AgentMessageEvent(
                agent_name=event.agent.name,
                content=event.ai_message.content
            )
        )

    async def on_user_message(self, event: UserMessage):
        self.log.events.append(
            UserMessageEvent(
                content=event.content
            )
        )

    async def on_orchestrator_message(self, event: OrchestratorMessage):
        self.log.events.append(
            OrchestratorMessageEvent(
                content=event.content
            )
        )

    async def on_tool_call(self, event: ToolCall):
        self.log.events.append(
            ToolCallEvent(
                tool_name=event.tool_name
            )
        )

    async def on_tool_result(self, event: ToolResult):
        self.log.events.append(
            ToolResultEvent(
                tool_name=event.tool_name
                result = event.result
            )
        )

    async def on_orchestrator_start(self, event: OrchestratorStart):
        self.log.events.append(
            OrchestratorStartEvent()
        )

    async def on_orchestrator_end(self, event: OrchestratorEnd):
        self.log.events.append(
            OrchestratorEndEvent()
        )

    async def on_agent_message_delta(self, event: AgentMessageDelta):
        self.log.events.append(
            AgentMessageDeltaEvent(
                delta=event.delta
            )
        )

    async def on_orchestrator_error(self, event: OrchestratorError):
        self.log.events.append(
            OrchestratorErrorEvent(
                error=str(event.error)
            )
        )

    async def on_end_turn(self, event: EndTurn):
        self.log.events.append(
            EndTurnEvent()
        )

    async def on_event(self, event: Event):
        """
        Fallback for unhandled events if you want to store them too,
        or do something generic. The default Handler base class calls
        this method for all events. We can ignore or store it.
        """
        # self.log.events.append( ...some default model... )
        pass

    def get_log(self) -> EventsLog:
        """
        Expose the Pydantic log container for consumption.
        """
        return self.log