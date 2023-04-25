import prodigy
from prodigy.components.loaders import Images

OPTIONS = [
    {"id": 1, "text": "SERIOUS"},
    {"id": 2, "text": "SAD"},
    {"id": 3, "text": "GLAD"},
]

@prodigy.recipe("classify-images")
def classify_images(dataset, source):
    def get_stream():
        stream = Images(source)
        for example in stream:
            example["options"] = OPTIONS
            yield example

    return {
        "dataset": dataset,
        "stream": get_stream(),
        "view_id": "choice",
        "config": {
            "choice_style": "single",
            "choice_auto_accept": True
        }
    }
