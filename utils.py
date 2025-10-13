import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def visualize_dinov3_dense_features(
    features, original_image, patch_size=16, save_prefix="dinov3"
):
    """
    Visualize DINOv3 dense features using PCA→RGB mapping (as shown in DINOv3 Figure 13)

    This is the OFFICIAL method from the DINOv3 paper for visualizing dense features.

    Args:
        features: (num_patches, feature_dim) numpy array from DINOv3
        original_image: PIL Image or numpy array
        patch_size: 14 for ViT-*/14 or 16 for ViT-*/16
        save_prefix: prefix for saved images

    The visualization projects dense outputs using PCA and maps the first 3
    components to RGB channels, as done in DINOv3 Figure 13.
    """

    print("\n" + "=" * 70)
    print("DINOv3 DENSE FEATURE VISUALIZATION (Figure 13 Method)")
    print("=" * 70)

    # Get spatial dimensions
    num_patches = features.shape[0]
    feature_dim = features.shape[1]

    # Calculate feature map size
    H = W = int(np.sqrt(num_patches))

    if H * W != num_patches:
        print(f"⚠️  WARNING: Non-square feature map detected")
        H = int(np.sqrt(num_patches))
        W = num_patches // H

    print(f"\n📊 Feature Map Info:")
    print(f"  Feature map size: {H}×{W} (patches)")
    print(f"  Feature dimension: {feature_dim}")
    print(f"  Patch size: {patch_size}×{patch_size} pixels")
    print(f"  Approximate resolution: {H*patch_size}×{W*patch_size} pixels")

    # Quality checks
    if np.isnan(features).any() or np.isinf(features).any():
        print("  ❌ ERROR: NaN or Inf values detected!")
        return None
    print("  ✓ No NaN/Inf values")

    # STEP 1: Apply PCA to project dense outputs to 3D
    print(f"\n🎨 Applying PCA projection (DINOv3 official method):")
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features)  # (num_patches, 3)

    print(f"  PC1 variance: {pca.explained_variance_ratio_[0]:.1%}")
    print(f"  PC2 variance: {pca.explained_variance_ratio_[1]:.1%}")
    print(f"  PC3 variance: {pca.explained_variance_ratio_[2]:.1%}")
    print(f"  Total variance captured: {pca.explained_variance_ratio_.sum():.1%}")

    # STEP 2: Map PCA components to RGB
    # Normalize each component to [0, 1] independently
    pca_rgb = np.zeros_like(pca_features)
    for i in range(3):
        component = pca_features[:, i]
        pca_rgb[:, i] = (component - component.min()) / (
            component.max() - component.min() + 1e-8
        )

    # Reshape to spatial grid
    pca_image = pca_rgb.reshape(H, W, 3)

    # STEP 3: Additional quality metrics
    print(f"\n📈 Feature Quality Metrics:")

    # Feature magnitude
    feature_norms = np.linalg.norm(features, axis=1)
    print(f"  Mean L2 norm: {feature_norms.mean():.4f} ± {feature_norms.std():.4f}")

    # Cosine similarity statistics (to check feature diversity)
    center_idx = (H // 2) * W + (W // 2)
    center_feature = features[center_idx : center_idx + 1]
    similarities = cosine_similarity(center_feature, features)[0]
    print(
        f"  Center patch similarity - Mean: {similarities.mean():.3f}, Std: {similarities.std():.3f}"
    )

    if similarities.std() < 0.1:
        print("  ⚠️  WARNING: Low similarity diversity - features may be too uniform")
    elif similarities.mean() > 0.9:
        print(
            "  ⚠️  WARNING: Very high average similarity - features may lack distinctiveness"
        )
    else:
        print("  ✓ Good feature diversity")

    # STEP 4: Create visualization (matching DINOv3 Figure 13 style)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image", fontsize=14, fontweight="bold", pad=10)
    axes[0].axis("off")

    # PCA→RGB visualization (THE KEY VISUALIZATION FROM FIGURE 13)
    axes[1].imshow(pca_image, interpolation="bilinear")
    axes[1].set_title(
        f"Dense Features (PCA→RGB)\n"
        f"Feature map: {H}×{W} | Variance: {pca.explained_variance_ratio_.sum():.1%}",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )
    axes[1].axis("off")

    # Feature magnitude heatmap (additional quality check)
    feature_norms_map = feature_norms.reshape(H, W)
    im = axes[2].imshow(feature_norms_map, cmap="viridis", interpolation="bilinear")
    axes[2].set_title("Feature Magnitude", fontsize=14, fontweight="bold", pad=10)
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(
        "DINOv3 Dense Feature Visualization",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    # Save
    save_path = f"debug/{save_prefix}_pca_image.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n💾 Saved: {save_path}")
    plt.show()

    # STEP 5: Additional detailed analysis plot
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))

    # Individual PCA components
    for i in range(3):
        component = pca_features[:, i].reshape(H, W)
        im = axes2[0, i].imshow(component, cmap="RdBu_r", interpolation="bilinear")
        axes2[0, i].set_title(
            f"PC{i+1} ({pca.explained_variance_ratio_[i]:.1%})",
            fontsize=12,
            fontweight="bold",
        )
        axes2[0, i].axis("off")
        plt.colorbar(im, ax=axes2[0, i], fraction=0.046)

    # RGB channels separately
    for i, (channel, color) in enumerate(
        zip(["R", "G", "B"], ["Reds", "Greens", "Blues"])
    ):
        rgb_channel = pca_rgb[:, i].reshape(H, W)
        im = axes2[1, i].imshow(rgb_channel, cmap=color, interpolation="bilinear")
        axes2[1, i].set_title(
            f"{channel} Channel (PC{i+1})", fontsize=12, fontweight="bold"
        )
        axes2[1, i].axis("off")
        plt.colorbar(im, ax=axes2[1, i], fraction=0.046)

    plt.suptitle("PCA Component Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()

    save_path2 = f"debug/{save_prefix}_pca_components.png"
    plt.savefig(save_path2, dpi=300, bbox_inches="tight")
    print(f"💾 Saved: {save_path2}")
    plt.show()

    print("=" * 70 + "\n")

    # Assessment
    print("✅ QUALITY ASSESSMENT:")
    quality_score = 0

    if not (np.isnan(features).any() or np.isinf(features).any()):
        print("  ✓ No numerical issues")
        quality_score += 1

    if 0.5 <= pca.explained_variance_ratio_.sum() <= 0.95:
        print("  ✓ Good PCA variance distribution")
        quality_score += 1

    if 0.1 <= similarities.std() <= 0.4:
        print("  ✓ Good feature diversity")
        quality_score += 1

    if 0.3 <= similarities.mean() <= 0.8:
        print("  ✓ Good similarity range")
        quality_score += 1

    if quality_score >= 3:
        print(f"\n🎉 OVERALL: Good extraction quality ({quality_score}/4)")
    else:
        print(f"\n⚠️  OVERALL: Extraction may have issues ({quality_score}/4)")

    print()

    return {
        "pca_image": pca_image,
        "pca": pca,
        "pca_features": pca_features,
        "feature_norms": feature_norms_map,
        "quality_score": quality_score,
    }


def compare_feature_quality(features_list, labels, original_image):
    """
    Compare multiple feature extractions side-by-side (like Figure 13)

    Args:
        features_list: list of (num_patches, feature_dim) arrays
        labels: list of strings (model/layer names)
        original_image: PIL Image
        patch_size: patch size
    """
    n_features = len(features_list)
    fig, axes = plt.subplots(1, n_features + 1, figsize=(6 * (n_features + 1), 6))

    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Each feature visualization
    for idx, (features, label) in enumerate(zip(features_list, labels)):
        H = W = int(np.sqrt(features.shape[0]))

        # PCA→RGB
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(features)
        pca_rgb = np.zeros_like(pca_features)

        for i in range(3):
            component = pca_features[:, i]
            pca_rgb[:, i] = (component - component.min()) / (
                component.max() - component.min() + 1e-8
            )

        pca_image = pca_rgb.reshape(H, W, 3)

        axes[idx + 1].imshow(pca_image, interpolation="bilinear")
        axes[idx + 1].set_title(
            f"{label}\n{H}×{W} | Var: {pca.explained_variance_ratio_.sum():.1%}",
            fontsize=14,
            fontweight="bold",
        )
        axes[idx + 1].axis("off")

    plt.suptitle(
        "Dense Features Comparison (DINOv3)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("debug/feature_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
