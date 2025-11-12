#!/usr/bin/env python3
"""Test script to verify the edit test dialog fix."""

import sys
from PyQt6.QtWidgets import QApplication
from gui.rule_editor_dialog import _TestEditorDialog

def test_dialog():
    """Test that the dialog doesn't close when changing test types."""
    app = QApplication(sys.argv)
    
    # Create a test dialog
    dialog = _TestEditorDialog(
        parent=None,
        initial={"type": "snippet_detected", "name": "test"},
        snippet_names=["test", "example"],
        snippet_positions={"test": (100, 100, 50, 50)},
        snippet_swipes={}
    )
    
    # Test changing to different test types
    print("Testing dialog with different test types...")
    
    # Test snippet types
    dialog._kind_combo.setCurrentText("snippet_detected")
    print(f"snippet_detected - Dialog visible: {dialog.isVisible()}")
    
    dialog._kind_combo.setCurrentText("snippet_not_detected")
    print(f"snippet_not_detected - Dialog visible: {dialog.isVisible()}")
    
    # Test state types - this should not close the dialog
    dialog._kind_combo.setCurrentText("state_text_equals")
    print(f"state_text_equals - Dialog visible: {dialog.isVisible()}")
    
    dialog._kind_combo.setCurrentText("player_state_is")
    print(f"player_state_is - Dialog visible: {dialog.isVisible()}")
    
    dialog._kind_combo.setCurrentText("custom_expression")
    print(f"custom_expression - Dialog visible: {dialog.isVisible()}")
    
    # Show the dialog for manual testing
    dialog.show()
    print("\nDialog is displayed. Try changing test types manually.")
    print("The dialog should NOT close when changing types.")
    
    return app.exec()

if __name__ == "__main__":
    test_dialog()