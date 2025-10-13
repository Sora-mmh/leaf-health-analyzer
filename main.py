import os
from pprint import pprint
from analyzer import DINOv3PlantHealthAnalyzer

if __name__ == "__main__":
    token = os.environ.get("H_HUB_TOKEN")
    analyzer = DINOv3PlantHealthAnalyzer(model_size="base", token=token)

    # Analyze single leaf
    result = analyzer.analyze_leaf_health("examples/Potato___Early_blight.png")
    pprint(result, indent=2, width=80)

    # Batch analysis
    # image_paths = ['leaf1.jpg', 'leaf2.jpg', 'leaf3.jpg']
    # results = analyzer.batch_analyze(image_paths)
