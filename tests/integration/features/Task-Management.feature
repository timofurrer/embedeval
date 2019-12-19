Feature: Built-in Tasks
    In order for the User to get started
    quickly, embedeval has built-in Tasks.
    Those tasks can be listed and new
    ones can be created.

    This Feature covers the Requirements: M4, M9, M11, M12

    Scenario: List all built-in Tasks
        Given the embedeval tool is installed
        When the tasks are listed with the CLI
        Then at least two built-in Tasks are available

    Scenario: Create new Task based on the template
        Given the embedeval tool is installed
        When a new Task is created in the "tasks/" directory with the CLI
        Then the new Task is available
        And the Task Python Module is based on the template

    Scenario: Create new Task based on existing Task
        Given the embedeval tool is installed
        When a new Task is created based on an existing Task in the "tasks/" directory with the CLI
        Then the new Task is available
        And the Task Python Module is based on the chosen existing Task
