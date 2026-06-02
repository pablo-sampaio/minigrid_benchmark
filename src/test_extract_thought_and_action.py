import unittest

from react_agent import extract_thought_and_action


class ExtractThoughtAndActionTests(unittest.TestCase):
    def test_xml_basic(self):
        response = """
<thought>
I should move forward.
</thought>
<action>
MOVE_FORWARD
</action>
"""
        thought, action = extract_thought_and_action(response)
        self.assertEqual(thought, "I should move forward.")
    self.assertEqual(action, "MOVE_FORWARD")

    def test_xml_action_with_trailing_punctuation(self):
        response = """
<thought>Turn now.</thought>
<action>gira_horario!!!</action>
"""
        thought, action = extract_thought_and_action(response)
        self.assertEqual(thought, "Turn now.")
        self.assertEqual(action, "TURN_RIGHT")

    def test_legacy_thought_and_action_standard(self):
        response = "THOUGHT: I am facing the goal.\nACTION: MOVE_FORWARD"
        thought, action = extract_thought_and_action(response)
        self.assertEqual(thought, "I am facing the goal.")
        self.assertEqual(action, "MOVE_FORWARD")

    def test_legacy_portuguese_with_accents(self):
        response = "PENSAMENTO: Vou girar para alinhar.\nAÇÃO: GIRA_ANTI_HORÁRIO"
        thought, action = extract_thought_and_action(response)
        self.assertEqual(thought, "Vou girar para alinhar.")
        self.assertEqual(action, "TURN_LEFT")

    def test_legacy_with_markdown_bold_labels(self):
        response = "**THOUGHT**: keep right\n**ACTION**: turn_right"
        thought, action = extract_thought_and_action(response)
        self.assertEqual(thought, "keep right")
        self.assertEqual(action, "TURN_RIGHT")

    def test_legacy_with_extra_spaces_and_newline_after_colon(self):
        response = "THOUGHT   :   adjust direction\nACTION:\n   MOVE_FORWARD"
        thought, action = extract_thought_and_action(response)
        self.assertEqual(thought, "adjust direction")
        self.assertEqual(action, "MOVE_FORWARD")

    def test_action_not_found_returns_empty_action(self):
        response = "THOUGHT: move toward target\nACTION: PULAR"
        thought, action = extract_thought_and_action(response)
        self.assertEqual(thought, "move toward target")
        self.assertEqual(action, "")

    def test_thought_not_found_returns_default(self):
        response = "ACTION: MOVE_FORWARD"
        thought, action = extract_thought_and_action(response)
        self.assertEqual(thought, "(not found)")
        self.assertEqual(action, "MOVE_FORWARD")

    def test_xml_has_priority_over_legacy(self):
        response = """
THOUGHT: legacy thought
ACTION: TURN_RIGHT
<thought>xml thought</thought>
<action>MOVE_FORWARD</action>
"""
        thought, action = extract_thought_and_action(response)
        self.assertEqual(thought, "xml thought")
        self.assertEqual(action, "MOVE_FORWARD")

    def test_xml_portuguese_action_is_canonicalized_to_english(self):
        response = """
<thought>Virar.</thought>
<action>GIRA_ANTI_HORARIO</action>
"""
        thought, action = extract_thought_and_action(response)
        self.assertEqual(thought, "Virar.")
        self.assertEqual(action, "TURN_LEFT")

    def test_empty_response(self):
        thought, action = extract_thought_and_action("")
        self.assertEqual(thought, "(not found)")
        self.assertEqual(action, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
