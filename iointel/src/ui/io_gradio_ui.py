import gradio as gr
import uuid
from iointel.src.utilities.formatting import format_result_for_html


class IOGradioUI:
    def __init__(self, agent, interface_title=None):
        self.agent = agent
        self.interface_title = (
            interface_title or f"Agent: {getattr(agent, 'name', 'Agent')}"
        )

    def launch(self, share=False):
        agent = self.agent
        interface_title = self.interface_title

        # Register the set_css tool (could be replaced with an Agent in the future)
        if not hasattr(agent, "_set_css_registered"):
            from iointel import register_tool

            register_tool(agent._set_css)
            agent._set_css_registered = True

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
        ):
            # If no conversation_id, generate one
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            # If user_message is None and dynamic_ui_values is not None, treat as dynamic UI submission
            if (
                user_message is None
                and dynamic_ui_spec is not None
                and dynamic_ui_values is not None
            ):
                # Send the dynamic UI values as the next message/tool call
                user_message = f"[DYNAMIC_UI_SUBMIT] {dynamic_ui_values}"
            # Run the agent asynchronously and append the result to the chat
            result = await agent.run(
                user_message, conversation_id=conversation_id, pretty=False
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
            # Inject favicon for IO.net branding
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
                conv_id = conv_id_input
            with gr.Row(elem_id="input-row"):
                user_input = gr.Textbox(label="Your message", scale=4)
                send_btn = gr.Button("Send", scale=1, elem_classes=["io-chat-btn"])
            # Hidden HTML for dynamic CSS injection
            css_html = gr.HTML(f"<style>{default_css}</style>", visible=False)
            # Column for dynamic UI
            dynamic_ui_col = gr.Column(visible=False)
            dynamic_ui_inputs = []

            def render_dynamic_ui(ui_spec):
                # Clear previous UI
                dynamic_ui_col.children = []
                dynamic_ui_inputs.clear()
                if ui_spec:
                    for comp in ui_spec:
                        if comp["type"] == "textbox":
                            tb = gr.Textbox(
                                label=comp.get("label", ""), value=comp.get("value", "")
                            )
                            dynamic_ui_col.append(tb)
                            dynamic_ui_inputs.append(tb)
                        elif comp["type"] == "slider":
                            sl = gr.Slider(
                                minimum=comp.get("min", 0),
                                maximum=comp.get("max", 100),
                                value=comp.get("value", 0),
                                label=comp.get("label", ""),
                            )
                            dynamic_ui_col.append(sl)
                            dynamic_ui_inputs.append(sl)
                    submit_btn = gr.Button(
                        "Submit Dynamic UI", elem_classes=["io-chat-btn"]
                    )
                    dynamic_ui_col.append(submit_btn)
                    dynamic_ui_col.visible = True

                    def on_dynamic_submit(*values):
                        # Pass these values as dynamic_ui_values to agent_chat_fn
                        return agent_chat_fn(
                            chatbot.value,
                            None,
                            conv_id_state.value,
                            css_html.value,
                            ui_spec,
                            list(values),
                        )

                    submit_btn.click(
                        on_dynamic_submit,
                        inputs=dynamic_ui_inputs,
                        outputs=[
                            chatbot,
                            user_input,
                            conv_id_state,
                            css_html,
                            dynamic_ui_spec_state,
                            dynamic_ui_values_state,
                        ],
                    )
                else:
                    dynamic_ui_col.visible = False

            conv_id_state = gr.State("")
            dynamic_ui_spec_state = gr.State(None)
            dynamic_ui_values_state = gr.State(None)
            # Set initial states on load
            demo.load(
                get_initial_states,
                inputs=None,
                outputs=[
                    chatbot,
                    conv_id_state,
                    css_html,
                    dynamic_ui_spec_state,
                    dynamic_ui_values_state,
                ],
            )

            def update_conv_id(cid):
                return cid

            conv_id.change(update_conv_id, inputs=conv_id, outputs=conv_id_state)
            # Main chat send
            send_btn.click(
                agent_chat_fn,
                inputs=[
                    chatbot,
                    user_input,
                    conv_id_state,
                    css_html,
                    dynamic_ui_spec_state,
                    dynamic_ui_values_state,
                ],
                outputs=[
                    chatbot,
                    user_input,
                    conv_id_state,
                    css_html,
                    dynamic_ui_spec_state,
                    dynamic_ui_values_state,
                ],
            )
            user_input.submit(
                agent_chat_fn,
                inputs=[
                    chatbot,
                    user_input,
                    conv_id_state,
                    css_html,
                    dynamic_ui_spec_state,
                    dynamic_ui_values_state,
                ],
                outputs=[
                    chatbot,
                    user_input,
                    conv_id_state,
                    css_html,
                    dynamic_ui_spec_state,
                    dynamic_ui_values_state,
                ],
            )
            # Dynamic UI rendering logic (pseudo):
            # If dynamic_ui_spec_state is not None, render components in dynamic_ui_col and show it
            # When user submits dynamic UI, call agent_chat_fn with user_message=None and dynamic_ui_values set
            render_dynamic_ui(dynamic_ui_spec_state.value)

        demo.launch(share=share)
