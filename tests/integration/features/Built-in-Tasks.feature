Feature: Built-in Tasks
    In order for the User to get started
    quickly, embedeval has built-in Tasks.

    This Feature covers the Requirements: M9, M11, M12

    Scenario: List all built-in Tasks
        Given the embedeval tool is installed
        When the tasks are listed with the CLI
        Then at least two built-in Tasks are available
