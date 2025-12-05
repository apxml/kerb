Fine-Tuning Module
==================

.. automodule:: kerb.fine_tuning
   :members:
   :undoc-members:
   :show-inheritance:

Model fine-tuning utilities and large dataset preparation.

Dataset Functions
-----------------

.. autofunction:: kerb.fine_tuning.create_training_dataset

.. autofunction:: kerb.fine_tuning.to_openai_format

.. autofunction:: kerb.fine_tuning.to_anthropic_format

JSONL Functions
---------------

.. autofunction:: kerb.fine_tuning.write_jsonl

.. autofunction:: kerb.fine_tuning.read_jsonl

.. autofunction:: kerb.fine_tuning.append_jsonl

.. autofunction:: kerb.fine_tuning.merge_jsonl

.. autofunction:: kerb.fine_tuning.validate_jsonl

.. autofunction:: kerb.fine_tuning.count_jsonl_lines

Dataset Classes
---------------

.. autoclass:: kerb.fine_tuning.TrainingExample
   :members:
   :undoc-members:

.. autoclass:: kerb.fine_tuning.TrainingDataset
   :members:
   :undoc-members:

.. autoclass:: kerb.fine_tuning.DatasetFormat
   :members:
   :undoc-members:
