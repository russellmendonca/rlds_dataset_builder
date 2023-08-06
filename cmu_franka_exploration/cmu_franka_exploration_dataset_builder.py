from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from PIL import Image

class CMUFrankaExploration(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset( {
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(64, 64, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                         'highres_image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='High resolution main camera observation',
                        )
                        # 'wrist_image': tfds.features.Image(
                        #     shape=(64, 64, 3),
                        #     dtype=np.uint8,
                        #     encoding_format='png',
                        #     doc='Wrist camera RGB observation.',
                        # ),
                        # 'state': tfds.features.Tensor(
                        #     shape=(10,),
                        #     dtype=np.float32,
                        #     doc='Robot state, consists of [7x robot joint angles, '
                        #         '2x gripper position, 1x door opening angle].',
                        # )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(8,),
                        dtype=np.float32,
                        doc='Robot action, consists of [end effector position3x, end effector orientation3x, gripper action1x, episode termination1x].',
                    ),
                    'structured_action': tfds.features.Tensor(
                        shape=(8,),
                        dtype=np.float32,
                        doc='Structured action, consisting of hybrid affordance and end-effector control, described in Structured World Models from Human Videos.',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='data/train/*.npz'),
            # 'val': self._generate_examples(path='data/val/*.npz'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case

            lang_instruction = data['language_instruction'].tolist()
            lang_embedding   = self._embed([lang_instruction])[0].numpy()
            
            #import ipdb ; ipdb.set_trace()
           
            episode = []
            ep_len = len(data['image'])
            for i in range(ep_len):

                highres_img = data['image'][i]
                image_64x64 = np.array(Image.fromarray(highres_img).resize((64,64), Image.Resampling.LANCZOS ))

                ep_dict = {
                    'observation': {
                        'image': image_64x64,
                        'highres_image' : highres_img
                    },
                    'action': data['action'].astype(np.float32)[i],
                    'structured_action': data['structured_action'].astype(np.float32)[i],
                    'discount': 1.0,
                    'reward':  data['reward'].astype(np.float32)[i],
                    'is_first': i == 0,
                    'is_last': i == (ep_len - 1),
                    'is_terminal': i == (ep_len - 1),
                    'language_instruction': lang_instruction,
                    'language_embedding': lang_embedding,
                }
                # if 'image_view2' in data:
                #     ep_dict['observation']['highres_image_view2'] = data['image_view2'][i]


                episode.append(ep_dict)

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

