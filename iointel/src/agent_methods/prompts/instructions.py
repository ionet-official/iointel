REASONING_INSTRUCTIONS = """
    You are working on solving a difficult problem (the `goal`). Based
    on your previous thoughts and the overall goal, please perform **one
    reasoning step** that advances you closer to a solution. Document
    your thought process and any intermediate steps you take.
    
    After marking this task complete for a single step, you will be
    given a new reasoning task to continue working on the problem. The
    loop will continue until you have a valid solution.
    
    Complete the task as soon as you have a valid solution.
    
    **Guidelines**
    
    - You will not be able to brute force a solution exhaustively. You
        must use your reasoning ability to make a plan that lets you make
        progress.
    - Each step should be focused on a specific aspect of the problem,
        either advancing your understanding of the problem or validating a
        solution.
    - You should build on previous steps without repeating them.
    - Since you will iterate your reasoning, you can explore multiple
        approaches in different steps.
    - Use logical and analytical thinking to reason through the problem.
    - Ensure that your solution is valid and meets all requirements.
    - If you find yourself spinning your wheels, take a step back and
        re-evaluate your approach.
"""
