import torch.nn as nn
from torchvision.models import efficientnet_b0


class DomainAdaptationModel(nn.Module):
    def __init__(self):
        super(DomainAdaptationModel, self).__init__()
        self.feature_extractor = efficientnet_b0(weights="IMAGENET1K_V1")
        feature_size = self.feature_extractor.classifier[1].in_features
        self.feature_extractor.classifier[1] = nn.Identity()

        self.label_classifier = nn.Linear(feature_size, 2)
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x, alpha=1.0, domain=False):
        features = self.feature_extractor(x)
        if domain:
            return self.domain_classifier(features)
        return self.label_classifier(features)
