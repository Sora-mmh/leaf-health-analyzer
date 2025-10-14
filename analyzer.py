from pathlib import Path
import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from scipy import ndimage
import cv2
from datetime import datetime
import warnings
from transformers import AutoImageProcessor, AutoModel

from utils import visualize_dinov3_dense_features, compare_feature_quality

DINOV3_REPO = "/home/mmhamdi/workspace/vlms/vege/dinov3"
sys.path.insert(0, DINOV3_REPO)

# from dinov3.eval.segmentation.models import (
#     build_segmentation_decoder,
#     BackboneLayersSet,
# )
from dinov3.eval.segmentation.inference import make_inference


warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class DINOv3PlantHealthAnalyzer:
    def __init__(self, token=None, mode="local", backbone_size=None):
        """
        Initialize DINOv3 backbone for plant health analysis using Transformers
        backbone_size: 'small', 'base', 'large', or 'giant'
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if mode != "local" and backbone_size is not None:
            # backbone name mapping for DINOv3 from Transformers
            backbone_names = {
                "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
                "base": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
                "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                "giant": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
            }
            backbone_name = backbone_names.get(
                backbone_size, "facebook/dinov3-vitb16-pretrain-lvd1689m"
            )
            # Load DINOv3 backbone and processor from Transformers
            print(f"Loading {backbone_name} backbone from Transformers...")
            self.processor = AutoImageProcessor.from_pretrained(
                backbone_name, token=token
            )
            self.backbone = AutoModel.from_pretrained(backbone_name, token=token)
            self.backbone.to(self.device)
            self.backbone.eval()
            print(f"Loaded DINOv3 ViT-{backbone_size.upper()} backbone successfully")

        else:
            self._build_segmentation_model()

        # Define health indicators based on feature patterns
        self.health_indicators = {
            "healthy": {"color_range": (0.3, 0.7), "texture_uniformity": 0.8},
            # "stressed": {"color_range": (0.5, 0.8), "texture_uniformity": 0.6},
            "diseased": {"color_range": (0.7, 1.0), "texture_uniformity": 0.4},
        }

    def _build_segmentation_model(self):
        BACKBONE_WEIGHTS = "/home/mmhamdi/workspace/vlms/vege/dinov3_weights/segmentation/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
        SEGMENTOR_WEIGHTS = "/home/mmhamdi/workspace/vlms/vege/dinov3_weights/segmentation/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth"
        # Load segmentor with backbone and head weights
        self.segmentor = torch.hub.load(
            DINOV3_REPO,
            "dinov3_vit7b16_ms",
            source="local",
            weights=SEGMENTOR_WEIGHTS,
            backbone_weights=BACKBONE_WEIGHTS,
        )

        # """Attach called DinoV3 backbone from HF with its segmentation head"""
        # checkpoint = torch.load(self.SEGMENTOR_HEAD_WEIGHTS, map_location="cpu")
        # self.segmentation_model = build_segmentation_decoder(
        #     backbone_model=self.backbone,
        #     backbone_out_layers=BackboneLayersSet.FOUR_EVEN_INTERVALS,
        #     decoder_type="m2f",
        #     hidden_dim=2048,
        #     num_classes=150,  # ADE20K
        #     autocast_dtype=(
        #         torch.bfloat16 if self.device.type == "cuda" else torch.float32
        #     ),
        # )
        # # Load the weights (the checkpoint contains the full model)
        # self.segmentation_model.load_state_dict(checkpoint, strict=False)
        self.segmetor = self.segmentor.to(self.device)
        # self.segmentation_model.eval()
        print(f"Loaded segmentation model.")

    def extract_leaf_features(self, image_path):
        """Extract dense features from leaf image using Transformers"""
        # Load image
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        self.img_name = Path(image_path).stem
        # Preprocess image using the processor
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        num_register_tokens = self.backbone.config.num_register_tokens

        segmentation_map = self.segmentation_model.predict(
            inputs["pixel_values"],
            rescale_to=(original_size[0], original_size[1]),
        )

        with torch.no_grad():
            # Get backbone outputs
            outputs = self.backbone(**inputs, output_hidden_states=True)

            # Extract features from multiple layers
            hidden_states = outputs.hidden_states

            # Get the last layer features (excluding CLS token)
            main_features = (
                hidden_states[-1][:, 1 + num_register_tokens :, :]
                .squeeze(0)
                .cpu()
                .numpy()
            )

            # Get earlier layer for texture analysis (layer 4)
            texture_features = (
                hidden_states[2][:, 1 + num_register_tokens :, :]
                .squeeze(0)
                .cpu()
                .numpy()
            )

            all_layers_features = [
                hidden_states[idx][:, 1 + num_register_tokens :, :]
                .squeeze(0)
                .cpu()
                .numpy()
                for idx in range(len(hidden_states))
            ]
            layers_labels = [f"Layer_{idx}" for idx in range(len(hidden_states))]
        visualize_dinov3_dense_features(
            main_features, image, save_prefix="main_features"
        )
        visualize_dinov3_dense_features(
            texture_features, image, save_prefix="texture_features"
        )
        compare_feature_quality(all_layers_features, layers_labels, image)
        # Store metadata
        patch_grid_size = int(np.sqrt(main_features.shape[0]))
        self.image_metadata = {
            "original_size": original_size,
            "patch_grid_size": patch_grid_size,
            "num_patches": main_features.shape[0],
        }

        return main_features, texture_features, image

    def segment_leaf_regions(self, features, image):
        """Segment leaf into healthy, and diseased regions"""
        patch_grid_size = self.image_metadata["patch_grid_size"]

        # Calculate feature statistics per patch
        feature_magnitude = np.linalg.norm(features, axis=1)
        feature_variance = np.var(features, axis=1)

        # Perform clustering to identify different leaf regions
        n_clusters = 2  # healthy, diseased
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features)

        # Reshape cluster labels to 2D
        cluster_map = cluster_labels.reshape(patch_grid_size, patch_grid_size)

        # Identify cluster types based on feature characteristics
        cluster_stats = {}
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_stats[i] = {
                "mean_magnitude": np.mean(feature_magnitude[cluster_mask]),
                "mean_variance": np.mean(feature_variance[cluster_mask]),
                "size": np.sum(cluster_mask),
            }

        # Sort clusters by magnitude (usually: background < healthy < diseased)
        sorted_clusters = sorted(
            cluster_stats.items(), key=lambda x: x[1]["mean_magnitude"]
        )

        # Map clusters to health categories
        health_map = np.zeros_like(cluster_map)
        if len(sorted_clusters) == 2:
            # # Assuming first cluster is background
            # background_id = sorted_clusters[0][0]
            # health_map[cluster_map == background_id] = 0  # Background
            # Map remaining clusters to health states
            healthy_id = sorted_clusters[0][0]
            health_map[cluster_map == healthy_id] = 0  # Healthy
            diseased_id = sorted_clusters[1][0]
            health_map[cluster_map == diseased_id] = 1  # Diseased

        return (
            health_map,
            cluster_stats,
            feature_magnitude.reshape(patch_grid_size, patch_grid_size),
        )

    def detect_disease_spots(self, features, texture_features, quant):
        """Detect specific disease spots or abnormalities"""
        patch_grid_size = self.image_metadata["patch_grid_size"]

        # Combine main and texture features for disease detection
        combined_features = np.concatenate([features, texture_features], axis=1)

        # Calculate anomaly scores based on local feature deviation
        feature_mean = np.mean(combined_features, axis=0)
        deviations = np.linalg.norm(combined_features - feature_mean, axis=1)

        # Threshold for anomalies (disease spots)
        threshold = np.percentile(deviations, quant)
        anomaly_mask = deviations > threshold

        # Reshape to 2D
        anomaly_map = anomaly_mask.reshape(patch_grid_size, patch_grid_size)
        deviation_map = deviations.reshape(patch_grid_size, patch_grid_size)

        # Find connected components (disease spots)
        labeled_spots, num_spots = ndimage.label(anomaly_map)

        disease_spots = []
        for i in range(1, num_spots + 1):
            spot_mask = labeled_spots == i
            spot_coords = np.argwhere(spot_mask)

            if len(spot_coords) > 2:  # Filter out very small spots
                center = np.mean(spot_coords, axis=0)
                size = len(spot_coords)
                max_deviation = np.max(deviation_map[spot_mask])

                disease_spots.append(
                    {
                        "center": center,
                        "size": size,
                        "severity": max_deviation,
                        "coords": spot_coords,
                    }
                )

        return disease_spots, anomaly_map, deviation_map

    def calculate_health_metrics(self, health_map, disease_spots, features):
        """Calculate comprehensive health metrics"""
        patch_grid_size = self.image_metadata["patch_grid_size"]
        total_patches = patch_grid_size * patch_grid_size

        # Exclude background (assuming it's labeled as 0)
        leaf_mask = health_map > 0
        leaf_area = np.sum(leaf_mask)

        if leaf_area == 0:
            return None

        # Calculate area percentages
        healthy_area = np.sum(health_map == 1) / leaf_area * 100
        # stressed_area = np.sum(health_map == 2) / leaf_area * 100
        diseased_area = np.sum(health_map == 2) / leaf_area * 100

        # Calculate disease severity
        disease_severity = 0
        if disease_spots:
            severities = [spot["severity"] for spot in disease_spots]
            disease_severity = np.mean(severities) if severities else 0

        # Calculate texture uniformity (healthy leaves have more uniform texture)
        feature_variance = np.var(features[leaf_mask.flatten()], axis=0)
        texture_uniformity = 1.0 / (1.0 + np.mean(feature_variance))

        # Calculate color consistency (using feature magnitudes as proxy)
        feature_magnitudes = np.linalg.norm(features, axis=1)
        color_consistency = 1.0 - (
            np.std(feature_magnitudes[leaf_mask.flatten()])
            / np.mean(feature_magnitudes[leaf_mask.flatten()])
        )

        # Calculate overall health score (0-100)
        health_score = (healthy_area * 1.0 + diseased_area * 0.0) * (
            texture_uniformity * 0.3
            + color_consistency * 0.3
            + (1 - min(disease_severity, 1)) * 0.4
        )

        metrics = {
            "health_score": min(max(health_score, 0), 100),
            "healthy_area_percentage": healthy_area,
            # "stressed_area_percentage": stressed_area,
            "diseased_area_percentage": diseased_area,
            "num_disease_spots": len(disease_spots),
            "disease_severity": disease_severity,
            "texture_uniformity": texture_uniformity,
            "color_consistency": color_consistency,
            "leaf_coverage": leaf_area / total_patches * 100,
        }

        return metrics

    def generate_health_report(self, metrics):
        """Generate detailed health report with recommendations"""
        if not metrics:
            return "Unable to analyze leaf health. Please ensure the image contains a clear leaf."

        score = metrics["health_score"]

        # Determine health category
        if score >= 80:
            category = "Excellent"
            color = "green"
        elif score >= 60:
            category = "Good"
            color = "yellowgreen"
        elif score >= 40:
            category = "Fair"
            color = "orange"
        elif score >= 20:
            category = "Poor"
            color = "darkorange"
        else:
            category = "Critical"
            color = "red"

        report = {
            "overall_status": category,
            "score": score,
            "color": color,
            "details": {
                "Healthy Tissue": f"{metrics['healthy_area_percentage']:.1f}%",
                # "Stressed Tissue": f"{metrics['stressed_area_percentage']:.1f}%",
                "Diseased Tissue": f"{metrics['diseased_area_percentage']:.1f}%",
                "Disease Spots": metrics["num_disease_spots"],
                "Texture Quality": f"{metrics['texture_uniformity']:.2f}",
                "Color Consistency": f"{metrics['color_consistency']:.2f}",
            },
        }

        # Add recommendations based on metrics
        recommendations = []

        if metrics["diseased_area_percentage"] > 20:
            recommendations.append(
                "• Consider fungicide treatment for disease management"
            )
        # if metrics["stressed_area_percentage"] > 30:
        #     recommendations.append("• Check watering schedule and nutrient levels")
        if metrics["num_disease_spots"] > 5:
            recommendations.append("• Isolate plant to prevent disease spread")
        if metrics["texture_uniformity"] < 0.5:
            recommendations.append("• Monitor for pest activity")
        if metrics["color_consistency"] < 0.6:
            recommendations.append("• Assess light exposure and nutrient balance")

        if score >= 80:
            recommendations.append("• Continue current care routine")
            recommendations.append("• Perform regular monitoring")

        report["recommendations"] = recommendations

        return report

    def visualize_health_analysis(
        self, image, health_map, disease_spots, anomaly_map, metrics, report
    ):
        """Comprehensive visualization of health analysis"""
        fig = plt.figure(figsize=(20, 12))

        # Original image
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(image)
        ax1.set_title("Original Leaf Image", fontsize=14, fontweight="bold")
        ax1.axis("off")

        # Health map
        ax2 = plt.subplot(2, 3, 2)
        health_colors = ["black", "green", "red"]
        cmap = plt.cm.colors.ListedColormap(health_colors)

        # Resize health map to match image dimensions
        original_width, original_height = self.image_metadata["original_size"]
        patch_grid_size = self.image_metadata["patch_grid_size"]

        health_map_resized = cv2.resize(
            health_map.astype(np.float32),
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )

        im2 = ax2.imshow(health_map_resized, cmap=cmap, vmin=0, vmax=3, alpha=0.8)
        ax2.imshow(image, alpha=0.3)
        ax2.set_title("Health Segmentation Map", fontsize=14, fontweight="bold")
        ax2.axis("off")

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            # Patch(facecolor="black", label="Background"),
            Patch(facecolor="green", label="Healthy"),
            # Patch(facecolor="yellow", label="Stressed"),
            Patch(facecolor="red", label="Diseased"),
        ]
        ax2.legend(handles=legend_elements, loc="upper right", fontsize=10)

        # Disease spots visualization
        ax3 = plt.subplot(2, 3, 3)
        ax3.imshow(image)

        if disease_spots:
            scale_x = original_width / patch_grid_size
            scale_y = original_height / patch_grid_size

            for spot in disease_spots:
                center = spot["center"] * [scale_y, scale_x]
                size = spot["size"] * 10
                severity_normalized = min(
                    spot["severity"] / max([s["severity"] for s in disease_spots]), 1.0
                )

                circle = plt.Circle(
                    (center[1], center[0]),
                    radius=np.sqrt(size),
                    color="red",
                    alpha=0.3 + severity_normalized * 0.4,
                    linewidth=2,
                    fill=False,
                )
                ax3.add_patch(circle)

                # Add severity label for major spots
                if spot["size"] > 5:
                    ax3.text(
                        center[1],
                        center[0],
                        f"{spot['severity']:.1f}",
                        color="white",
                        fontsize=8,
                        ha="center",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                    )

        ax3.set_title(
            f"Disease Spots Detection ({len(disease_spots)} spots)",
            fontsize=14,
            fontweight="bold",
        )
        ax3.axis("off")

        # Anomaly heatmap
        ax4 = plt.subplot(2, 3, 4)
        anomaly_resized = cv2.resize(
            anomaly_map.astype(np.float32),
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR,
        )
        im4 = ax4.imshow(anomaly_resized, cmap="hot", alpha=0.7)
        ax4.imshow(image, alpha=0.3)
        ax4.set_title("Anomaly Heatmap", fontsize=14, fontweight="bold")
        ax4.axis("off")
        plt.colorbar(im4, ax=ax4, label="Anomaly Level")

        # Health score gauge
        ax5 = plt.subplot(2, 3, 5)
        self.draw_health_gauge(
            ax5, report["score"], report["overall_status"], report["color"]
        )

        # Metrics and recommendations
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis("off")

        # Create text for metrics
        metrics_text = f"PLANT HEALTH REPORT\n" + "=" * 30 + "\n\n"
        metrics_text += f"Overall Status: {report['overall_status']}\n"
        metrics_text += f"Health Score: {report['score']:.1f}/100\n\n"

        metrics_text += "Tissue Composition:\n"
        for key, value in report["details"].items():
            metrics_text += f"  {key}: {value}\n"

        metrics_text += "\nRecommendations:\n"
        for rec in report["recommendations"]:
            metrics_text += f"{rec}\n"

        ax6.text(
            0.1,
            0.9,
            metrics_text,
            transform=ax6.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.suptitle(
            f'Plant Health Analysis - {datetime.now().strftime("%Y-%m-%d %H:%M")}',
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        plt.savefig(
            f"outputs/{self.img_name}_health_analysis.png", dpi=300, bbox_inches="tight"
        )
        # plt.show()

        return fig

    def draw_health_gauge(self, ax, score, status, color):
        """Draw a circular gauge for health score"""
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        radius_outer = 1.0
        radius_inner = 0.6

        # Draw gauge background
        for i, (t1, t2) in enumerate(zip(theta[:-1], theta[1:])):
            if i < 20:
                c = "red"
            elif i < 40:
                c = "darkorange"
            elif i < 60:
                c = "orange"
            elif i < 80:
                c = "yellowgreen"
            else:
                c = "green"

            x = [
                radius_inner * np.cos(t1),
                radius_outer * np.cos(t1),
                radius_outer * np.cos(t2),
                radius_inner * np.cos(t2),
            ]
            y = [
                radius_inner * np.sin(t1),
                radius_outer * np.sin(t1),
                radius_outer * np.sin(t2),
                radius_inner * np.sin(t2),
            ]

            ax.fill(x, y, color=c, alpha=0.3)

        # Draw needle
        needle_angle = np.pi * (1 - score / 100)
        needle_length = 0.9
        ax.arrow(
            0,
            0,
            needle_length * np.cos(needle_angle),
            needle_length * np.sin(needle_angle),
            head_width=0.1,
            head_length=0.1,
            fc="black",
            ec="black",
            linewidth=3,
        )

        # Draw center circle
        circle = plt.Circle((0, 0), 0.15, color="black", zorder=10)
        ax.add_patch(circle)

        # Add score text
        ax.text(
            0,
            -0.3,
            f"{score:.0f}",
            ha="center",
            va="center",
            fontsize=28,
            fontweight="bold",
            color=color,
        )
        ax.text(
            0,
            -0.5,
            status,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color=color,
        )

        # Set axis properties
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.7, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Health Score", fontsize=14, fontweight="bold", pad=20)

    def batch_analyze(self, image_paths):
        """Analyze multiple leaf images and compare health"""
        results = []

        for path in image_paths:
            print(f"\nAnalyzing: {path}")
            result = self.analyze_leaf_health(path, show_visualization=False)
            if result:
                results.append(
                    {
                        "path": path,
                        "metrics": result["metrics"],
                        "report": result["report"],
                    }
                )

        if results:
            # Create comparison visualization
            self.compare_results(results)

        return results

    def compare_results(self, results):
        """Create comparison visualization for multiple leaves"""
        fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))

        if len(results) == 1:
            axes = [axes]

        for ax, result in zip(axes, results):
            score = result["report"]["score"]
            status = result["report"]["overall_status"]
            color = result["report"]["color"]

            self.draw_health_gauge(ax, score, status, color)
            ax.set_title(f"{result['path'].split('/')[-1][:20]}...", fontsize=10)

        plt.suptitle("Leaf Health Comparison", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(
            f"outputs/{self.img_name}_health_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        # plt.show()

    def analyze_leaf_health(self, image_path, show_visualization=True):
        """
        Complete analysis pipeline for leaf health
        """
        print("Extracting features from leaf image...")
        features, texture_features, image = self.extract_leaf_features(image_path)

        print("Segmenting leaf regions...")
        health_map, cluster_stats, feature_magnitude = self.segment_leaf_regions(
            features, image
        )

        print("Detecting disease spots...")
        disease_spots, anomaly_map, deviation_map = self.detect_disease_spots(
            features, texture_features, quant=60
        )

        print(f"Found {len(disease_spots)} potential disease spots")

        print("Calculating health metrics...")
        metrics = self.calculate_health_metrics(health_map, disease_spots, features)

        if metrics is None:
            print(
                "Could not calculate health metrics. Ensure image shows a leaf clearly."
            )
            return None

        print("Generating health report...")
        report = self.generate_health_report(metrics)

        print(f"\n{'='*50}")
        print(f"HEALTH ANALYSIS COMPLETE")
        print(f"{'='*50}")
        print(f"Overall Health Status: {report['overall_status']}")
        print(f"Health Score: {report['score']:.1f}/100")
        print(f"Healthy Tissue: {metrics['healthy_area_percentage']:.1f}%")
        # print(f"Stressed Tissue: {metrics['stressed_area_percentage']:.1f}%")
        print(f"Diseased Tissue: {metrics['diseased_area_percentage']:.1f}%")
        print(f"Disease Spots: {metrics['num_disease_spots']}")
        print(f"{'='*50}")

        if show_visualization:
            # Visualize results
            fig = self.visualize_health_analysis(
                image, health_map, disease_spots, anomaly_map, metrics, report
            )

        return {
            "metrics": metrics,
            "report": report,
            "health_map": health_map,
            "disease_spots": disease_spots,
            "cluster_stats": cluster_stats,
        }
