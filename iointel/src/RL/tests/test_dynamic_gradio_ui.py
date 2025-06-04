import gradio as gr
import ast
from iointel.src.ui.dynamic_ui import MAX_TEXTBOXES, MAX_SLIDERS


def default_spec():
    return [
        {"type": "textbox", "label": "Favorite weekend activity", "value": ""},
        {
            "type": "slider",
            "label": "Preferred work hours per week",
            "min": 0,
            "max": 80,
            "value": 40,
        },
        {
            "type": "textbox",
            "label": "Last book or movie that inspired you",
            "value": "",
        },
        {
            "type": "slider",
            "label": "Number of close friends",
            "min": 0,
            "max": 50,
            "value": 5,
        },
    ]


with gr.Blocks() as demo:
    gr.Markdown("# Dynamic UI Test (Production Logic)")
    spec_input = gr.Textbox(
        label="Dynamic UI Spec (Python list of dicts)",
        value=str(default_spec()),
        lines=6,
    )
    update_btn = gr.Button("Update UI Spec")
    dynamic_ui_col = gr.Column(visible=True)
    with dynamic_ui_col:
        predefined_textboxes = [gr.Textbox(visible=False) for _ in range(MAX_TEXTBOXES)]
        predefined_sliders = [gr.Slider(visible=False) for _ in range(MAX_SLIDERS)]
    dynamic_ui_inputs = []
    # State to hold current values for all dynamic UI components
    dynamic_values_state = gr.State(
        ["" for _ in range(MAX_TEXTBOXES)] + [0 for _ in range(MAX_SLIDERS)]
    )

    def update_ui(spec_str, current_values):
        try:
            ui_spec = ast.literal_eval(spec_str)
        except Exception as e:
            ui_spec = []
            print("[ERROR] Failed to parse UI spec:", e)
        # Build updates using current values
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

    update_btn.click(
        update_ui,
        inputs=[spec_input, dynamic_values_state],
        outputs=predefined_textboxes + predefined_sliders + [dynamic_values_state],
    )

    # Main input and submit for the whole form (like chat)
    main_input = gr.Textbox(label="Main Input (simulates chat box)")
    main_submit = gr.Button("Submit")
    output = gr.Textbox(label="Output")

    def collect_dynamic_ui_values(*args):
        # args: all textbox and slider values in order
        return list(args)

    def on_main_submit(main_text, *args):
        values = collect_dynamic_ui_values(*args)
        # Example: map values to labels using a default spec (or pass the spec as needed)
        # value_map = map_dynamic_ui_values_to_labels(default_spec(), values)
        print("[TEST] Main input:", main_text, "Dynamic UI values:", values)
        return f"Main input: {main_text}\nDynamic UI values: {values}"

    main_submit.click(
        on_main_submit,
        inputs=[main_input] + predefined_textboxes + predefined_sliders,
        outputs=output,
    )

demo.launch()
