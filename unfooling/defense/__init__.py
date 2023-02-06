import re

from unfooling.defense.vae import VAE
from unfooling.defense.cad import GMMCADFull
from unfooling.defense.cknn import KNNCAD
from unfooling.defense.cad_classification import CADClassificationDetector
from unfooling.defense.novelty import NoveltyDetector
from unfooling.defense.nb import NaiveBayes


def get_detector(detector_name, hparams, problem=None):
    if detector_name == 'CAD-GMM':
        detector = GMMCADFull(**hparams)
    elif re.search(r'^CAD-CLF($|-.)', detector_name):
        if detector_name == 'CAD-CLF':
            kwargs = {}
        else:
            # CAD-CLF-{GMM,VAE,LOF,IF}
            kwargs = {'name': detector_name.split('CAD-CLF-', 1)[-1]}
        detector = CADClassificationDetector(
            **hparams,
            **kwargs,
        )
    elif detector_name == 'CVAE':
        detector = VAE(**hparams)
    elif detector_name == 'KNNCAD':
        detector = KNNCAD(**hparams)
    elif detector_name == 'NB':
        detector = NaiveBayes(**hparams)
    else:
        selected_idxs = None
        if hparams.get('strategy') in {'selected_idxs',
                                       'selected_idxs_independent'}:
            assert problem is not None, (
                f'problem must be provided when strategy={hparams["strategy"]}')
            selected_idxs = problem.sensitive_feature_idxs
            print(f'OOD strategy for sensitive features: '
                  f'{problem.sensitive_features}')

        kwargs = ({} if 'name' in hparams else
                  {'name': re.sub(r'-IND$', '', detector_name)})
        detector = NoveltyDetector(
            n_jobs=-1,
            selected_idxs=selected_idxs,
            **hparams,
            **kwargs,
        )

    return detector
