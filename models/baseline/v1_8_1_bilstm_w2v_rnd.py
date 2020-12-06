
from torch import nn

from qiqc.config import ExperimentConfigBuilderBase
# from qiqc.modules import BinaryClassifier
# from qiqc.presets.v1_8_1_bilsm_w2v_rnd import TextNormalizerPresets
# from qiqc.presets.v1_8_1_bilsm_w2v_rnd import TextTokenizerPresets
# from qiqc.presets.v1_8_1_bilsm_w2v_rnd import WordEmbeddingFeaturizerPresets
# from qiqc.presets.v1_8_1_bilsm_w2v_rnd import WordExtraFeaturizerPresets
# from qiqc.presets.v1_8_1_bilsm_w2v_rnd import SentenceExtraFeaturizerPresets
# from qiqc.presets.v1_8_1_bilsm_w2v_rnd import PreprocessorPresets
# from qiqc.presets.v1_8_1_bilsm_w2v_rnd import EmbeddingPresets
# from qiqc.presets.v1_8_1_bilsm_w2v_rnd import EncoderPresets
# from qiqc.presets.v1_8_1_bilsm_w2v_rnd import AggregatorPresets
# from qiqc.presets.v1_8_1_bilsm_w2v_rnd import MLPPresets


#from qiqc.presets.v1_8_1_bilsm_w2v_rnd import EnsemblerPresets


# =======  Experiment configuration  =======

class ExperimentConfigBuilder(ExperimentConfigBuilderBase):

    default_config = dict(
        test=False,
        device=None,
        maxlen=72,
        vocab_mincount=5,
        scale_batchsize=[],
        validate_from=2,
    )

    @property
    def modules(self):
        return [
            TextNormalizer,
            TextTokenizer,
            WordEmbeddingFeaturizer,
            WordExtraFeaturizer,
            SentenceExtraFeaturizer,
            Embedding,
            Encoder,
            Aggregator,
            MLP,
        ]

# =======  Preprocessing modules  =======

class TextNormalizer(TextNormalizerPresets):
    pass


class TextTokenizer(TextTokenizerPresets):
    pass


class WordEmbeddingFeaturizer(WordEmbeddingFeaturizerPresets):
    pass


class WordExtraFeaturizer(WordExtraFeaturizerPresets):
    pass


class SentenceExtraFeaturizer(SentenceExtraFeaturizerPresets):
    pass


class Preprocessor(PreprocessorPresets):
    pass


# =======  Training modules  =======

class Embedding(EmbeddingPresets):
    pass


class Encoder(EncoderPresets):
    pass


class Aggregator(AggregatorPresets):
    pass


class MLP(MLPPresets):
    pass


class Ensembler(EnsemblerPresets):
    pass