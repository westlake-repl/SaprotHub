import sys

root_dir = __file__.rsplit("/", 2)[0]
if root_dir not in sys.path:
    sys.path.append(root_dir)

import gradio as gr

from utils import set_text_bg_color
from loop_retrieve_cards import get_models, get_datasets, get_readme_dict


def match_card(input: str, card_id: str, card_type: str) -> str:
    """
    Search the input in a card. If the input string is contained in the card_id or its README, display this card.

    Args:
        input:  Input string
        card_id: HuggingFace card id
        card_type: Type of card, either "model" or "dataset"
    """
    display_str = ""
    readme_dict = get_readme_dict()

    if input.lower() in card_id.lower() or input.lower() in readme_dict[card_id].lower():
        # Add card id
        if card_type == "model":
            display_str += f"## [{set_text_bg_color(input, card_id)}](https://huggingface.co/{card_id})\n\n"
        else:
            display_str += f"## [{set_text_bg_color(input, card_id)}](https://huggingface.co/datasets/{card_id})\n\n"

        # Highlight lines that contain the input string
        show_lines = []
        for line in readme_dict[card_id].split("\n"):
            if input.lower() in line.lower() and "<!--" not in line:
                show_lines.append(set_text_bg_color(input, line))

        # Add README
        display_str += "\n\n".join(show_lines)

        # Add a separator
        display_str = f"\n\n{display_str}\n\n---\n\n"

        # In case that the keyword is only contained in comments
        if input.lower() not in card_id.lower() and len(show_lines) == 0:
            display_str = ""

    return display_str


def show_card_info(input: str):
    retrieval_str = ""

    if input != "":
        # Search models
        retrieval_str += "# Models\n\n"
        for model in get_models():
            retrieval_str += match_card(input, model, "model")

        # Search datasets
        retrieval_str += "# Datasets\n\n"
        for dataset in get_datasets():
            retrieval_str += match_card(input, dataset, "dataset")

    return gr.Markdown(retrieval_str, visible=True)


# Build demo
with gr.Blocks(title="SaprotHub", fill_width=True) as demo:
    gr.Label("SaprotHub search", visible=True, show_label=False)
    search_box = gr.Textbox(label="Search box", placeholder="Input keywords to search", interactive=True, scale=0, container=True)

    # Display search results
    search_hint = gr.Markdown("# Search results:", visible=True)
    items = gr.Markdown(visible=False)

    # Set events
    search_box.change(show_card_info, inputs=[search_box], outputs=[items])


if __name__ == '__main__':
    # Run the demo
    demo.launch(server_name="0.0.0.0")
