import gradio as gr

MAX_TEXTBOXES = 10
MAX_SLIDERS = 10


def map_dynamic_ui_values_to_labels(spec, values):
    """
    Given a UI spec and a list of values, return a dict mapping labels to values.
    """
    result = {}
    tb_idx = 0
    sl_idx = 0
    for comp in spec or []:
        if comp["type"] == "textbox":
            result[comp.get("label", f"Textbox {tb_idx + 1}")] = values[tb_idx]
            tb_idx += 1
        elif comp["type"] == "slider":
            result[comp.get("label", f"Slider {sl_idx + 1}")] = values[
                MAX_TEXTBOXES + sl_idx
            ]
            sl_idx += 1
    return result


def get_dynamic_ui_updates(ui_spec, predefined_textboxes, predefined_sliders):
    updates = []
    tb_idx = 0
    sl_idx = 0
    # First, fill in the textboxes in order of appearance in the spec
    for comp in ui_spec or []:
        if comp["type"] == "textbox" and tb_idx < MAX_TEXTBOXES:
            updates.append(
                gr.update(
                    label=comp.get("label", f"Textbox {tb_idx + 1}"), visible=True
                )
            )
            tb_idx += 1
    # Hide unused textboxes
    for i in range(tb_idx, MAX_TEXTBOXES):
        updates.append(gr.update(visible=False))
    # Now, fill in the sliders in order of appearance in the spec
    for comp in ui_spec or []:
        if comp["type"] == "slider" and sl_idx < MAX_SLIDERS:
            updates.append(
                gr.update(
                    label=comp.get("label", f"Slider {sl_idx + 1}"),
                    minimum=comp.get("min", 0),
                    maximum=comp.get("max", 100),
                    visible=True,
                )
            )
            sl_idx += 1
    # Hide unused sliders
    for i in range(sl_idx, MAX_SLIDERS):
        updates.append(gr.update(visible=False))
    return updates
