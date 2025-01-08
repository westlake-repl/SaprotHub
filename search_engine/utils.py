import requests
import re


def fetch_models(author: str = "SaProtHub") -> list:
    """
    Retrieve models belonging to a specific author

    Args:
        author: Author name

    Returns:
        models: List of models
    """

    url = f"https://hf-mirror.com/api/models?author={author}"
    response = requests.get(url)
    models_dict = response.json()
    models = [item["id"] for item in models_dict]

    return models


def fetch_datasets(author: str = "SaProtHub") -> list:
    """
    Retrieve datasets belonging to a specific author

    Args:
        author: Author name

    Returns:
        datasets: List of datasets
    """

    url = f"https://hf-mirror.com/api/datasets?author={author}"
    response = requests.get(url)
    datasets_dict = response.json()
    datasets = [item["id"] for item in datasets_dict]

    return datasets


def fetch_readme(card_id: str, card_type: str) -> str:
    """
    Retrieve the README file of a model or dataset

    Args:
        card_id: Model or dataset ID
        card_type: Type of card, either "model" or "dataset"

    Returns:
        readme: README text
    """
    if card_type == "model":
        url = f"https://hf-mirror.com/{card_id}/raw/main/README.md"
    else:
        url = f"https://hf-mirror.com/datasets/{card_id}/raw/main/README.md"

    response = requests.get(url)
    readme = response.text.split("---")[-1]

    return readme


def set_text_bg_color(pattern: str, text: str, color: str = "yellow") -> str:
    """
    Set the background color of a pattern in a text

    Args:
        pattern: Pattern to highlight
        text: Text to search
        color: Background color

    Returns:
        text: Text with highlighted pattern
    """

    # Find all matches
    matches = set(re.findall(re.escape(pattern), text, flags=re.IGNORECASE))
    if len(matches) == 0:
        # No matches found
        return text

    replace_dict = {re.escape(m): f'<span style="background-color:{color}">{m}</span>' for m in matches}
    pattern = re.compile("|".join(replace_dict.keys()))
    text = pattern.sub(lambda m: replace_dict[re.escape(m.group(0))], text)

    return text

