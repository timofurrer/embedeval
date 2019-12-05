Feature: Evaluation Report for a German Word Embedding
    In order to allow german-speaking Users
    to evaluate their German Word Embedding
    embedeval sohuld be able to generate
    reports for them.

    This Feature covers the Requirements: M5, M6, M7

    Scenario: Evaluate and report on a German Word Embedding
        Given the Word Embedding "german.vec"
        When the Word Embedding is evaluated with the "odd-one-out" Task
        Then the Task should pass
        And the Task should generate a report highlighting the odd word
