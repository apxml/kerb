Safety Module
=============

.. automodule:: kerb.safety
   :members:
   :undoc-members:
   :show-inheritance:

Content moderation and safety filters.

Safety Functions
----------------

.. autofunction:: kerb.safety.moderate_content

.. autofunction:: kerb.safety.check_toxicity

.. autofunction:: kerb.safety.detect_pii

.. autofunction:: kerb.safety.redact_pii

.. autofunction:: kerb.safety.detect_prompt_injection

Safety Classes
--------------

.. autoclass:: kerb.safety.ModerationResult
   :members:
   :undoc-members:

.. autoclass:: kerb.safety.SafetyResult
   :members:
   :undoc-members:

.. autoclass:: kerb.safety.PIIMatch
   :members:
   :undoc-members:

.. autoclass:: kerb.safety.Guardrail
   :members:
   :undoc-members:
