import gradio as gr
import uuid
from iointel.src.ui.formatting import format_result_for_html
from iointel.src.ui.dynamic_ui import (
    MAX_TEXTBOXES,
    MAX_SLIDERS,
    map_dynamic_ui_values_to_labels,
)
import ast
import json


class IOGradioUI:
    def __init__(self, agent, interface_title=None):
        self.agent = agent
        self.interface_title = (
            interface_title or f"Agent: {getattr(agent, 'name', 'Agent')}"
        )

    def launch(self, share=False):
        agent = self.agent
        interface_title = self.interface_title

        convos = agent.get_conversation_ids()
        if convos:
            conv_id_input = gr.Dropdown(
                choices=convos,
                label="Conversation ID",
                value=convos[0] if convos else "",
            )
        else:
            conv_id_input = gr.Textbox(label="Conversation ID", value="", visible=True)

        default_css = """
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
        .tool-pill {
            background: #18181b;
            border: 1.5px solid #ffb300;
            border-radius: 10px;
            margin: 10px 0;
            padding: 10px 16px;
            box-shadow: 0 2px 12px #0004;
        }
        .tool-pill pre {
            background: #23272f;
            color: #ffb300;
            border-radius: 6px;
            padding: 4px 8px;
            font-size: 0.98em;
            box-shadow: 0 2px 8px #0002;
        }
        .input-row {
            position: sticky;
            bottom: 0;
            background: #18181b;
            z-index: 10;
            padding-bottom: 12px;
            border-radius: 0 0 12px 12px;
            box-shadow: 0 -2px 16px #0005;
        }
        .gradio-container input, .gradio-container textarea {
            background: #23272f;
            color: #fff;
            border: 1.5px solid #333;
            border-radius: 8px;
            padding: 10px;
            font-size: 1em;
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

        def get_initial_states():
            # history, conversation_id, css, dynamic_ui_spec, dynamic_ui_values
            return [], "", f"<style>{default_css}</style>", None, None

        async def agent_chat_fn(
            history,
            user_message,
            conversation_id,
            css,
            dynamic_ui_spec,
            dynamic_ui_values,
            dynamic_ui_history,
        ):
            if not conversation_id:
                conversation_id = str(uuid.uuid4())

            # If user_message is None and dynamic_ui_values is not None, treat as dynamic UI submission
            if (
                user_message is None
                and dynamic_ui_spec is not None
                and dynamic_ui_values is not None
            ):
                value_map = map_dynamic_ui_values_to_labels(
                    dynamic_ui_spec, dynamic_ui_values
                )
                user_message = (
                    f"[DYNAMIC_UI_SUBMIT] {json.dumps(value_map, ensure_ascii=False)}"
                )

            # Serialize the dynamic_ui_history
            history_str = json.dumps(dynamic_ui_history, ensure_ascii=False)
            # Combine with user_message
            combined_message = (
                f"DYNAMIC_UI_HISTORY: {history_str}\nUSER: {user_message}"
            )
            # Run the agent asynchronously and append the result to the chat
            result = await agent.run(
                combined_message, conversation_id=conversation_id, pretty=True
            )
            # Check for set_css tool usage
            for tur in result.get("tool_usage_results", []):
                if getattr(tur, "tool_name", "") == "_set_css":
                    css = tur.tool_result.get("css", css)
            # Check for gradio_dynamic_ui tool usage
            dynamic_ui_spec = None
            for tur in result.get("tool_usage_results", []):
                if getattr(tur, "tool_name", "") == "gradio_dynamic_ui":
                    if tur.tool_result is not None:
                        dynamic_ui_spec = tur.tool_result.get("ui", None)
            agent_html = (
                f'<div class="agent-bubble">{format_result_for_html(result)}</div>'
            )
            history = history or []
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": agent_html})
            return (
                history,
                "",
                conversation_id,
                f"<style>{css}</style>",
                dynamic_ui_spec,
                None,
            )

        io_favicon_url = "https://io.net/favicon.ico"
        favicon_html = gr.HTML(
            f"""
        <link rel='icon' type='image/x-icon' href='{io_favicon_url}' />
        """,
            visible=True,
        )

        with gr.Blocks() as demo:
            favicon_html.render()
            gr.Markdown(f"# {interface_title}")
            with gr.Row():
                chatbot = gr.Chatbot(
                    label="Chat",
                    elem_id="chatbot",
                    show_copy_button=True,
                    height=600,
                    type="messages",
                )
                conv_id_input
            # Place the dynamic UI column directly in the layout so it always renders
            dynamic_ui_col = gr.Column(visible=True)
            with dynamic_ui_col:
                predefined_textboxes = [
                    gr.Textbox(visible=False) for _ in range(MAX_TEXTBOXES)
                ]
                predefined_sliders = [
                    gr.Slider(visible=False) for _ in range(MAX_SLIDERS)
                ]
            with gr.Row(elem_id="input-row"):
                user_input = gr.Textbox(label="Ask IO", scale=4)
                send_btn = gr.Button("Send", scale=1, elem_classes=["io-chat-btn"])
            css_html = gr.HTML(f"<style>{default_css}</style>", visible=False)
            conv_id_state = gr.State("")
            dynamic_ui_spec_state = gr.State(None)
            dynamic_ui_values_state = gr.State(
                ["" for _ in range(MAX_TEXTBOXES)] + [0 for _ in range(MAX_SLIDERS)]
            )
            dynamic_ui_history_state = gr.State([])

            # Wire up dynamic UI components to update state as user interacts
            def update_dynamic_ui_value(idx, is_slider=False):
                def _update(val, values):
                    new_values = list(values)
                    new_values[idx] = val
                    return new_values

                return _update

            for i, tb in enumerate(predefined_textboxes):
                tb.input(
                    update_dynamic_ui_value(i, is_slider=False),
                    inputs=[tb, dynamic_ui_values_state],
                    outputs=dynamic_ui_values_state,
                )
            for i, sl in enumerate(predefined_sliders):
                sl.change(
                    update_dynamic_ui_value(MAX_TEXTBOXES + i, is_slider=True),
                    inputs=[sl, dynamic_ui_values_state],
                    outputs=dynamic_ui_values_state,
                )

            # When the dynamic UI spec changes, update the UI using gr.update and state
            def update_dynamic_ui_callback(ui_spec, current_values):
                updates = []
                tb_idx = 0
                sl_idx = 0
                values = current_values or (
                    ["" for _ in range(MAX_TEXTBOXES)] + [0 for _ in range(MAX_SLIDERS)]
                )
                # Textboxes
                for comp in ui_spec or []:
                    if comp["type"] == "textbox" and tb_idx < MAX_TEXTBOXES:
                        updates.append(
                            gr.update(
                                label=comp.get("label", f"Textbox {tb_idx + 1}"),
                                value=values[tb_idx],
                                visible=True,
                            )
                        )
                        tb_idx += 1
                for i in range(tb_idx, MAX_TEXTBOXES):
                    updates.append(gr.update(visible=False))
                # Sliders
                for comp in ui_spec or []:
                    if comp["type"] == "slider" and sl_idx < MAX_SLIDERS:
                        updates.append(
                            gr.update(
                                label=comp.get("label", f"Slider {sl_idx + 1}"),
                                minimum=comp.get("min", 0),
                                maximum=comp.get("max", 100),
                                value=values[MAX_TEXTBOXES + sl_idx],
                                visible=True,
                            )
                        )
                        sl_idx += 1
                for i in range(sl_idx, MAX_SLIDERS):
                    updates.append(gr.update(visible=False))
                return (*updates, values)

            # Connect the callback to dynamic_ui_spec_state changes
            dynamic_ui_spec_state.change(
                update_dynamic_ui_callback,
                inputs=[dynamic_ui_spec_state, dynamic_ui_values_state],
                outputs=predefined_textboxes
                + predefined_sliders
                + [dynamic_ui_values_state],
            )

            # Function to render a static, read-only block for the dynamic UI in chat history
            def format_dynamic_ui_for_chat(ui_spec, values):
                if not ui_spec or not values:
                    return ""
                html = "<div class='agent-bubble'><b>Dynamic UI Submission:</b><br>"
                tb_idx = 0
                sl_idx = 0
                for comp in ui_spec or []:
                    if comp["type"] == "textbox" and tb_idx < MAX_TEXTBOXES:
                        html += f"<div><b>{comp.get('label', 'Textbox')}:</b> {values[tb_idx]}</div>"
                        tb_idx += 1
                    elif comp["type"] == "slider" and sl_idx < MAX_SLIDERS:
                        html += f"<div><b>{comp.get('label', 'Slider')}:</b> {values[MAX_TEXTBOXES + sl_idx]}</div>"
                        sl_idx += 1
                html += "</div>"
                return html

            async def chat_with_dynamic_ui(
                chatbot_val,
                user_input_val,
                conv_id_val,
                css_html_val,
                dynamic_ui_spec_val,
                dynamic_ui_values_val,
                dynamic_ui_history_val,
            ):
                history = chatbot_val or []
                new_dynamic_ui_history = list(dynamic_ui_history_val)
                print("[DEBUG] dynamic_ui_spec_val:", dynamic_ui_spec_val)
                print("[DEBUG] dynamic_ui_values_val:", dynamic_ui_values_val)
                if dynamic_ui_spec_val and dynamic_ui_values_val:
                    print("[DEBUG] Adding static UI block to chat history.")
                    static_ui_html = format_dynamic_ui_for_chat(
                        dynamic_ui_spec_val, dynamic_ui_values_val
                    )
                    history = history + [{"role": "system", "content": static_ui_html}]
                    new_dynamic_ui_history.append(
                        {
                            "spec": dynamic_ui_spec_val,
                            "values": list(dynamic_ui_values_val),
                        }
                    )
                result = await agent_chat_fn(
                    history,
                    user_input_val,
                    conv_id_val,
                    css_html_val,
                    dynamic_ui_spec_val,
                    dynamic_ui_values_val,
                    new_dynamic_ui_history,
                )
                return (*result, new_dynamic_ui_history, new_dynamic_ui_history)

            with gr.Accordion("Debug Info", open=False):
                debug_dynamic_ui_history = gr.JSON(label="Dynamic UI History (debug)")

            send_btn.click(
                chat_with_dynamic_ui,
                inputs=[
                    chatbot,
                    user_input,
                    conv_id_state,
                    css_html,
                    dynamic_ui_spec_state,
                    dynamic_ui_values_state,
                    dynamic_ui_history_state,
                ],
                outputs=[
                    chatbot,
                    user_input,
                    conv_id_state,
                    css_html,
                    dynamic_ui_spec_state,
                    dynamic_ui_values_state,
                    dynamic_ui_history_state,
                    debug_dynamic_ui_history,
                ],
            )
            user_input.submit(
                chat_with_dynamic_ui,
                inputs=[
                    chatbot,
                    user_input,
                    conv_id_state,
                    css_html,
                    dynamic_ui_spec_state,
                    dynamic_ui_values_state,
                    dynamic_ui_history_state,
                ],
                outputs=[
                    chatbot,
                    user_input,
                    conv_id_state,
                    css_html,
                    dynamic_ui_spec_state,
                    dynamic_ui_values_state,
                    dynamic_ui_history_state,
                    debug_dynamic_ui_history,
                ],
            )

        demo.launch(share=share)


def on_main_submit(main_text, *args, context, spec_input):
    try:
        ui_spec = ast.literal_eval(spec_input)
    except Exception:
        ui_spec = []
    new_context = context + [{"spec": ui_spec, "values": list(args)}]
    print(
        "[TEST] Main input:",
        main_text,
        "Dynamic UI values:",
        list(args),
        "Context:",
        new_context,
    )
    return (
        f"Main input: {main_text}\nDynamic UI values: {list(args)}\nContext: {new_context}",
        new_context,
    )
