from dash import Dash, html, dcc, callback, Output, Input 
from plotly import express, graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import os

import my_sam2
from my_sam_manager import MySamManager


# print("Generating masks.....")
# t1 = time.perf_counter()
# masks = my_sam2.generate_masks(image)
# t2 = time.perf_counter()
# print(f"Done generating masks - took {t2-t1} sec")
# mask_image = my_sam2.generate_mask_image(masks)

storage = r"labels/"  # TODO: CHANGE ME
images_dir = r"data/Dataset/Igneous/Basalt"  # TODO: CHANGE ME
images = ({"image": np.array(plt.imread(file)), "filename": file.name} for file in os.scandir(images_dir))

    
# image_src = r"data/Dataset/Igneous/Basalt/6.jpg"
# image = np.array(plt.imread(image_src))
image_obj = next(images)
sam_manager = MySamManager(image_obj, images_dir, storage)
# mask_images = list()
# mask_image_bool = np.zeros((image.shape[:-1]), dtype=bool)

fig = {
    "data": [go.Image(z=sam_manager.image)]
}
# cur_choice = "Add"
center_text = {"textAlign": "center"}

app = Dash()
app.layout = [ 
    html.H1(children="Segment Any Rocks (SAR)", style=center_text),
    # dcc.RadioItems(id="choice", options=["Add", "Remove"], value=cur_choice),
    dcc.Graph(id="image", figure=fig),
    html.Div([html.Button("Next", id="next_button", n_clicks=None),
              html.Button("Save", id="save_button", n_clicks=None)], style=center_text),
    html.P(id="info", children="Begin", style=center_text)
]

# @callback(Input("choice", "value"))
# def update_choice(choice):
#     global cur_choice
#     cur_choice = choice
@callback(
    Output("image", "figure", allow_duplicate=True),
    Output("info", "children", allow_duplicate=True),
    Input("next_button", "n_clicks"),
    prevent_initial_call=True
)
def next_image(_):
    image_obj = next(images, None)
    if image_obj is not None:
        global sam_manager
        sam_manager = MySamManager(image_obj, images_dir, storage)
        fig["data"] = [go.Image(z=sam_manager.image)]
    return fig, "Begin"


@callback(
    Output("info", "children"),
    Input("save_button", "n_clicks"),
    prevent_initial_call=True
)
def save_masks(n_clicks):
    sam_manager.persist()
    return "Saved!"

@callback(
    Output("image", "figure"),
    # Output("info", "children"),
    Input("image", "clickData"),
    prevent_initial_call=True
)
def segment(click_data):
    info = click_data["points"][0]
    x, y = info["x"], info["y"]
    # if (cur_choice == "Add" and not exists) or (cur_choice == "Remove" and exists):
    if not sam_manager.exists(x, y):
        _masks, _scores, _ = my_sam2.predict(sam_manager.image, np.array([[x, y]]), np.array([1]), multimask_output=False)
        _apply_segment(_masks[0])
    return fig

def _apply_segment(mask):
    # Track which region has already been processed
    _mask_bool = np.bool(mask)
    sam_manager.update_mask(_mask_bool)
    sam_manager.store_mask(mask)  # masks - label data
    
    display_mask = go.Image(z=_generate_display_mask(sam_manager.mask_image), opacity=0.3)
    
    if len(fig["data"]) > 1:
        fig["data"][1] = display_mask
    else:
        fig["data"].append(display_mask)

def _generate_display_mask(mask_bool):
    display_mask = sam_manager.image.copy()
    display_mask[mask_bool] = my_sam2.generate_mask_image(mask_bool)[mask_bool]
    return display_mask


if __name__ == "__main__":
    app.run(debug=True)