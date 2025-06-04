import gradio as gr
import uuid
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Callable

from iointel.src.ui.formatting import format_result_for_html
from iointel.src.ui.dynamic_ui import (
    MAX_TEXTBOXES,
    MAX_SLIDERS,
    map_dynamic_ui_values_to_labels,
)

DEFAULT_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
body, .gradio-container {
    font-family: 'Inter', sans-serif;
    background: #111 !important;
    color: #fff;
}
#chatbot {
    height: 600px !important;
    overflow-y: auto;
    background: #18181b;
    border-radius: 12px;
    box-shadow: 0 4px 32px #000a;
    padding: 8px 0;
}
.user-bubble {
    background: #2563eb;
    color: #fff;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 18px;
    margin: 8px 0;
    max-width: 80%;
    align-self: flex-end;
    font-size: 1.05em;
    box-shadow: 0 2px 8px #0003;
}
.agent-bubble {
    background: #23272f;
    color: #fff;
    border-radius: 16px 16px 16px 4px;
    padding: 12px 18px;
    margin: 8px 0;
    max-width: 80%;
    align-self: flex-start;
    font-size: 1.05em;
    box-shadow: 0 2px 8px #0003;
}
.io-chat-btn {
    background: linear-gradient(90deg, #ffb300 0%, #ff8c00 100%);
    color: #18181b;
    border: none;
    border-radius: 8px;
    font-weight: 700;
    font-size: 1em;
    padding: 10px 20px;
    box-shadow: 0 2px 8px #0002;
    transition: background 0.2s;
}
.io-chat-btn:hover {
    background: linear-gradient(90deg, #ff8c00 0%, #ffb300 100%);
}
"""

@dataclass
class ChatResponse:
    history: List[Dict[str, str]]
    new_message: str
    conv_id: str
    css: str
    dynamic_ui_spec: Optional[List[Dict[str, Any]]]
    dynamic_ui_values: Optional[List[Any]]


class IOGradioUI:
    def __init__(self, agent, interface_title: Optional[str] = None):
        self.agent = agent
        self.interface_title = interface_title or f"Agent: {getattr(agent, 'name', 'Agent')}"

    def _format_dynamic_ui_for_chat(self, ui_spec, values) -> str:
        if not ui_spec or not values:
            return ""
        html = "<div class='agent-bubble'><b>Dynamic UI Submission:</b><br>"
        tb_idx = sl_idx = 0
        for comp in ui_spec:
            if comp["type"] == "textbox" and tb_idx < MAX_TEXTBOXES:
                html += f"<div><b>{comp.get('label', 'Textbox')}:</b> {values[tb_idx]}</div>"
                tb_idx += 1
            elif comp["type"] == "slider" and sl_idx < MAX_SLIDERS:
                html += f"<div><b>{comp.get('label', 'Slider')}:</b> {values[MAX_TEXTBOXES + sl_idx]}</div>"
                sl_idx += 1
        return html + "</div>"

    def _update_dynamic_ui_value(self, idx: int, is_slider: bool = False) -> Callable:
        def _update(val, values):
            new_values = list(values)
            new_values[idx] = val
            return new_values
        return _update

    def _update_dynamic_ui_components(self, ui_spec, current_values):
        updates = []
        tb_idx = sl_idx = 0
        values = current_values or (["" for _ in range(MAX_TEXTBOXES)] + [0 for _ in range(MAX_SLIDERS)])

        for comp in ui_spec or []:
            if comp["type"] == "textbox" and tb_idx < MAX_TEXTBOXES:
                updates.append(gr.update(
                    label=comp.get("label", f"Textbox {tb_idx + 1}"),
                    value=values[tb_idx],
                    visible=True,
                ))
                tb_idx += 1

        for i in range(tb_idx, MAX_TEXTBOXES):
            updates.append(gr.update(visible=False))

        for comp in ui_spec or []:
            if comp["type"] == "slider" and sl_idx < MAX_SLIDERS:
                updates.append(gr.update(
                    label=comp.get("label", f"Slider {sl_idx + 1}"),
                    minimum=comp.get("min", 0),
                    maximum=comp.get("max", 100),
                    value=values[MAX_TEXTBOXES + sl_idx],
                    visible=True,
                ))
                sl_idx += 1

        for i in range(sl_idx, MAX_SLIDERS):
            updates.append(gr.update(visible=False))

        return (*updates, values)

    async def _chat_logic(self, 
                          history: List[Dict[str, str]],
                          user_message: Optional[str],
                          conversation_id: str,
                          css: str,
                          dynamic_ui_spec: Optional[List[Dict[str, Any]]],
                          dynamic_ui_values: Optional[List[Any]],
                          dynamic_ui_history: List[Dict[str, Any]]) -> ChatResponse:

        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        if user_message is None and dynamic_ui_spec and dynamic_ui_values:
            value_map = map_dynamic_ui_values_to_labels(dynamic_ui_spec, dynamic_ui_values)
            user_message = f"[DYNAMIC_UI_SUBMIT] {json.dumps(value_map, ensure_ascii=False)}"

        combined_message = (
            f"DYNAMIC_UI_HISTORY: {json.dumps(dynamic_ui_history)}\nUSER: {user_message}"
        )

        result = await self.agent.run(combined_message, conversation_id=conversation_id, pretty=True)

        for tur in result.get("tool_usage_results", []):
            if getattr(tur, "tool_name", "") == "_set_css":
                css = tur.tool_result.get("css", css)

        new_ui_spec = None
        for tur in result.get("tool_usage_results", []):
            if getattr(tur, "tool_name", "") == "gradio_dynamic_ui":
                if tur.tool_result:
                    new_ui_spec = tur.tool_result.get("ui")

        agent_html = f'<div class="agent-bubble">{format_result_for_html(result)}</div>'
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": agent_html})

        return ChatResponse(
            history=history,
            new_message="",
            conv_id=conversation_id,
            css=f"<style>{css}</style>",
            dynamic_ui_spec=new_ui_spec,
            dynamic_ui_values=None
        )

    def launch(self, share: bool = False):
        with gr.Blocks(css=DEFAULT_CSS) as demo:
            gr.HTML("<link rel='icon' type='image/x-icon' href='https://io.net/favicon.ico' />")
            gr.Markdown(f"# {self.interface_title}")

            chatbot = gr.Chatbot(elem_id="chatbot", show_copy_button=True, height=600, type="messages")
            user_input = gr.Textbox(label="Ask IO", scale=4)
            send_btn = gr.Button("Send", scale=1, elem_classes=["io-chat-btn"])

            conv_id_state = gr.State("")
            dynamic_ui_values_state = gr.State(["" for _ in range(MAX_TEXTBOXES)] + [0 for _ in range(MAX_SLIDERS)])
            dynamic_ui_spec_state = gr.State(None)
            dynamic_ui_history_state = gr.State([])

            textboxes = [gr.Textbox(visible=False) for _ in range(MAX_TEXTBOXES)]
            sliders = [gr.Slider(visible=False) for _ in range(MAX_SLIDERS)]

            for i, tb in enumerate(textboxes):
                tb.input(self._update_dynamic_ui_value(i), inputs=[tb, dynamic_ui_values_state], outputs=dynamic_ui_values_state)
            for i, sl in enumerate(sliders):
                sl.change(self._update_dynamic_ui_value(MAX_TEXTBOXES + i, True), inputs=[sl, dynamic_ui_values_state], outputs=dynamic_ui_values_state)

            dynamic_ui_spec_state.change(
                self._update_dynamic_ui_components,
                inputs=[dynamic_ui_spec_state, dynamic_ui_values_state],
                outputs=textboxes + sliders + [dynamic_ui_values_state],
            )

            async def chat_handler(chatbot_val, user_input_val, conv_id_val, ui_spec, ui_vals, ui_hist):
                history = chatbot_val or []
                ui_hist = list(ui_hist or [])

                if ui_spec and ui_vals:
                    history.append({"role": "system", "content": self._format_dynamic_ui_for_chat(ui_spec, ui_vals)})
                    ui_hist.append({"spec": ui_spec, "values": list(ui_vals)})

                response = await self._chat_logic(history, user_input_val, conv_id_val, DEFAULT_CSS, ui_spec, ui_vals, ui_hist)

                return (
                    response.history,
                    response.new_message,
                    response.conv_id,
                    response.css,
                    response.dynamic_ui_spec,
                    response.dynamic_ui_values,
                    ui_hist,
                )

            send_btn.click(
                chat_handler,
                inputs=[chatbot, user_input, conv_id_state, dynamic_ui_spec_state, dynamic_ui_values_state, dynamic_ui_history_state],
                outputs=[chatbot, user_input, conv_id_state, dynamic_ui_spec_state, dynamic_ui_values_state, dynamic_ui_history_state],
            )
            user_input.submit(
                chat_handler,
                inputs=[chatbot, user_input, conv_id_state, dynamic_ui_spec_state, dynamic_ui_values_state, dynamic_ui_history_state],
                outputs=[chatbot, user_input, conv_id_state, dynamic_ui_spec_state, dynamic_ui_values_state, dynamic_ui_history_state],
            )

        demo.launch(share=share)
