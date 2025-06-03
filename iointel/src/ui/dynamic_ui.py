import gradio as gr

MAX_TEXTBOXES = 10
MAX_SLIDERS = 10

def create_dynamic_ui():
    predefined_textboxes = [gr.Textbox(visible=False, label=f"Textbox {i+1}") for i in range(MAX_TEXTBOXES)]
    predefined_sliders = [gr.Slider(visible=False, label=f"Slider {i+1}") for i in range(MAX_SLIDERS)]
    components = predefined_textboxes + predefined_sliders

    def render(ui_spec):
        tb_idx = 0
        sl_idx = 0
        for tb in predefined_textboxes:
            tb.visible = False
        for sl in predefined_sliders:
            sl.visible = False
        for comp in ui_spec or []:
            if comp["type"] == "textbox" and tb_idx < MAX_TEXTBOXES:
                tb = predefined_textboxes[tb_idx]
                tb.label = comp.get("label", f"Textbox {tb_idx+1}")
                tb.value = comp.get("value", "")
                tb.visible = True
                tb_idx += 1
            elif comp["type"] == "slider" and sl_idx < MAX_SLIDERS:
                sl = predefined_sliders[sl_idx]
                sl.label = comp.get("label", f"Slider {sl_idx+1}")
                sl.minimum = comp.get("min", 0)
                sl.maximum = comp.get("max", 100)
                sl.value = comp.get("value", 0)
                sl.visible = True
                sl_idx += 1
        # Hide unused
        for i in range(tb_idx, MAX_TEXTBOXES):
            predefined_textboxes[i].visible = False
        for i in range(sl_idx, MAX_SLIDERS):
            predefined_sliders[i].visible = False
        return components

    return components, render

def render_dynamic_ui(ui_spec, predefined_textboxes, predefined_sliders, dynamic_ui_col, dynamic_ui_inputs):
    print("[DEBUG] render_dynamic_ui called with:", ui_spec)
    # Hide all components initially
    for tb in predefined_textboxes:
        tb.visible = False
    for sl in predefined_sliders:
        sl.visible = False
    dynamic_ui_inputs.clear()
    if ui_spec:
        tb_idx = 0
        sl_idx = 0
        for comp in ui_spec:
            print("[DEBUG] Rendering component:", comp)
            if comp["type"] == "textbox" and tb_idx < MAX_TEXTBOXES:
                tb = predefined_textboxes[tb_idx]
                tb.label = comp.get("label", "")
                tb.value = comp.get("value", "")
                tb.visible = True
                dynamic_ui_inputs.append(tb)
                tb_idx += 1
            elif comp["type"] == "slider" and sl_idx < MAX_SLIDERS:
                sl = predefined_sliders[sl_idx]
                sl.label = comp.get("label", "")
                sl.minimum = comp.get("min", 0)
                sl.maximum = comp.get("max", 100)
                sl.value = comp.get("value", 0)
                sl.visible = True
                dynamic_ui_inputs.append(sl)
                sl_idx += 1
        dynamic_ui_col.visible = True
    else:
        print("[DEBUG] render_dynamic_ui: No ui_spec, hiding dynamic_ui_col")
        dynamic_ui_col.visible = False

# New function for Gradio update pattern
def get_dynamic_ui_updates(ui_spec, predefined_textboxes, predefined_sliders):
    updates = []
    tb_idx = 0
    sl_idx = 0
    # First, fill in the textboxes in order of appearance in the spec
    for comp in ui_spec or []:
        if comp["type"] == "textbox" and tb_idx < MAX_TEXTBOXES:
            updates.append(gr.update(
                label=comp.get("label", f"Textbox {tb_idx+1}"),
                visible=True
            ))
            tb_idx += 1
    # Hide unused textboxes
    for i in range(tb_idx, MAX_TEXTBOXES):
        updates.append(gr.update(visible=False))
    # Now, fill in the sliders in order of appearance in the spec
    for comp in ui_spec or []:
        if comp["type"] == "slider" and sl_idx < MAX_SLIDERS:
            updates.append(gr.update(
                label=comp.get("label", f"Slider {sl_idx+1}"),
                minimum=comp.get("min", 0),
                maximum=comp.get("max", 100),
                visible=True
            ))
            sl_idx += 1
    # Hide unused sliders
    for i in range(sl_idx, MAX_SLIDERS):
        updates.append(gr.update(visible=False))
    return updates 