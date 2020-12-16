from qiqc.registry import register_preprocessor

from _qiqc.preprocessing.modules.normalizers.rulebase import cylower

register_preprocessor('lower')(cylower)