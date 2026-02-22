"""
LLM-based Permutation Solver

This module provides functionality to use various LLM APIs (OpenAI, Gemini, Claude)
to generate algorithms for solving permutation puzzles using allowed moves (L, R, X).
"""

import os
import re
import json
import random
import tempfile
import subprocess
import sys
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PermutationConfig:
    """Configuration for permutation generation and moves."""
    length: int = 6
    moves: Dict[str, str] = None
    system_prompt: Optional[str] = None
    main_prompt_template: Optional[str] = None
    optimize_prompt_template: Optional[str] = None
    
    def __post_init__(self):
        if self.moves is None:
            self.moves = {
                'L': 'Left cyclic shift — shifts all elements one position to the left, with the first element moving to the end.',
                'R': 'Right cyclic shift — shifts all elements one position to the right, with the last element moving to the beginning.',
                'X': 'Transposition of the first two elements — swaps the elements at positions 0 and 1.'
            }


# =============================================================================
# Permutation Operations
# =============================================================================

def apply_L(arr: np.ndarray) -> np.ndarray:
    """Left cyclic shift: [0,1,2,3] -> [1,2,3,0]"""
    return np.roll(arr, -1)


def apply_R(arr: np.ndarray) -> np.ndarray:
    """Right cyclic shift: [0,1,2,3] -> [3,0,1,2]"""
    return np.roll(arr, 1)


def apply_X(arr: np.ndarray) -> np.ndarray:
    """Transpose first two elements: [0,1,2,3] -> [1,0,2,3]"""
    result = arr.copy()
    if len(result) >= 2:
        result[0], result[1] = result[1], result[0]
    return result


MOVE_FUNCTIONS = {
    'L': apply_L,
    'R': apply_R,
    'X': apply_X
}


def apply_moves(arr: np.ndarray, moves: List[str]) -> np.ndarray:
    """Apply a sequence of moves to an array."""
    result = arr.copy()
    for move in moves:
        if move in MOVE_FUNCTIONS:
            result = MOVE_FUNCTIONS[move](result)
        else:
            raise ValueError(f"Unknown move: {move}")
    return result


def generate_random_permutation(n: int) -> np.ndarray:
    """Generate a random permutation of [0, 1, ..., n-1]."""
    arr = np.arange(n)
    np.random.shuffle(arr)
    return arr


def is_trivial(arr: np.ndarray) -> bool:
    """Check if array is in trivial (sorted) state."""
    return np.array_equal(arr, np.arange(len(arr)))


# =============================================================================
# Prompts
# =============================================================================

SYSTEM_PROMPT = """You are an expert algorithm designer specializing in combinatorial puzzles and permutation theory.
Your task is to create efficient, constructive algorithms that solve permutation problems using only allowed moves.
You must provide working Python code that is self-contained and can be executed directly.
Always follow the constraints specified in the task and provide polynomial-time solutions."""

MAIN_PROMPT_TEMPLATE = """Task: Implement a constructive sorting algorithm that sorts a given vector using ONLY allowed moves (L, R, X).

Input: A vector a of length n (0-indexed) containing distinct integers from 0 to n-1.

Allowed moves:
L: Left cyclic shift — shifts all elements one position to the left, with the first element moving to the end. Example: [0,1,2,3] -> [1,2,3,0].
R: Right cyclic shift — shifts all elements one position to the right, with the last element moving to the beginning. Example: [0,1,2,3] -> [3,0,1,2].
X: Transposition of the first two elements — swaps the elements at positions 0 and 1. Example: [0,1,2,3] -> [1,0,2,3].

CRITICAL CONSTRAINTS:
1. NO BFS, DFS, or any graph search algorithms are allowed
2. The algorithm must run in POLYNOMIAL TIME (O(n^k) for some constant k)
3. No exponential-time algorithms (like brute force search through permutations)
4. Must use a constructive, iterative approach that builds the solution step by step
5. No storing or exploring multiple states simultaneously

Strict operational constraints:
- No other operations, slicing, built-in sorting functions, or creating new arrays are allowed (except for a copy to simulate sorting)
- All moves must be appended to the moves list immediately after performing them (as strings: 'L', 'R', or 'X')
- Applying the sequence of moves sequentially to a copy of the input vector must yield a fully sorted ascending array [0, 1, 2, ..., n-1]
- Moves can be used multiple times as needed
- The algorithm must continue applying moves until the array is fully sorted

ALGORITHMIC REQUIREMENTS:
- Use a constructive approach: develop a strategy that systematically brings elements to their correct positions
- Think in terms of bringing the smallest unsorted element to the front, then "locking" it in place
- Consider how L and R can help position elements for X swaps
- The solution should work for any n and have predictable, polynomial-time complexity

Expected approach types (choose one):
1. Adaptation of bubble sort/insertion sort using available moves
2. Strategy of bringing smallest element to front, then second smallest, etc.
3. Any other polynomial-time constructive approach

Implementation requirements:
- Implement a function solve(vector) that returns a tuple (moves, sorted_array):
    - moves: list of strings representing all moves performed (e.g., ['L', 'X', 'R', ...])
    - sorted_array: the final sorted array after applying all moves (as a list)
- Include CLI interface:
    - When script is executed directly, accept vector as command-line argument (parse sys.argv[1] as JSON)
    - Use {default_vector} as fallback if no arg is given
    - Output should be JSON object with keys "moves" and "sorted_array"
- Include minimal example in main block for quick testing
- Code must be fully self-contained and executable without external dependencies (only sys, json allowed)
- JSON output must always be structured and parseable for automated testing

Example usage:
    python solve_module.py "[3,1,2,0,4]"

Example output (for illustration):
{{
    "moves": ["X", "L", "R", "X"],
    "sorted_array": [0,1,2,3,4]
}}

IMPORTANT: Focus on developing a polynomial-time constructive algorithm, NOT graph search.
Provide ONLY the Python code, no explanations before or after."""


OPTIMIZE_PROMPT_TEMPLATE = """You are given a working Python algorithm that sorts a permutation using L, R, X moves.

Current algorithm (O(n³)):
```python
{algorithm_code}
```

YOUR TASK: Optimize to O(n²), specifically ≤ 0.5·n² + O(n) moves.

═══════════════════════════════════════════════════════════════
CRITICAL INSIGHT #1: CYCLE DECOMPOSITION APPROACH
═══════════════════════════════════════════════════════════════
Don't think "place element k at position k". Instead:
1. Decompose permutation into disjoint cycles
2. Each k-cycle needs exactly k-1 adjacent transpositions
3. Process each cycle by "rotating" elements within the cycle

Example: Permutation [2,3,1,4] has cycle (1→2→3→1)
This cycle needs 2 transpositions to resolve, NOT bubble-sorting each element.

═══════════════════════════════════════════════════════════════
CRITICAL INSIGHT #2: TRUE CHAINING (not fake chaining!)
═══════════════════════════════════════════════════════════════
WRONG (what your current code does):
  swap_at(i): L*i, X, R*i  # Cost: 2i+1 each, independent
  
RIGHT (true chaining for consecutive swaps at positions i < j < k):
  L*i, X, L*(j-i), X, L*(k-j), X, R*k  # Cost: 2k + 3 total!

The key: DON'T rotate back after each swap. Stay rotated and continue.

Concrete example for swaps at positions 2, 5, 7 (n=10):
  WRONG: (L L X R R) + (L L L L L X R R R R R) + (L*7 X R*7) = 5+11+15 = 31 moves
  RIGHT: L L X L L L X L L X R R R R R R R = 2+1+3+1+2+1+7 = 17 moves

═══════════════════════════════════════════════════════════════
CRITICAL INSIGHT #3: DIRECTION OPTIMIZATION
═══════════════════════════════════════════════════════════════
For position i:
- If i < n/2: use R*i X L*i (or chain with R direction)
- If i ≥ n/2: use L*(n-i) X R*(n-i) — going "the other way" is shorter!

This halves the rotation cost for positions in the second half.

═══════════════════════════════════════════════════════════════
CRITICAL INSIGHT #4: POST-PROCESSING
═══════════════════════════════════════════════════════════════
After generating moves, simplify:
- X.X → remove both (X is self-inverse)
- L.R or R.L → remove both
- Collapse L.L.L.R.R → L (net rotation)

═══════════════════════════════════════════════════════════════
CRITICAL INSIGHT #5: You must use chaining for BOTH halves of positions!
═══════════════════════════════════════════════════════════════
- Positions i < n/2: chain using L direction
- Positions i ≥ n/2: chain using R direction (going backwards)

DO NOT process swaps independently in a loop like:
  for pos in positions:
      swap_at(pos)  # WRONG - this is O(n) per swap!

═══════════════════════════════════════════════════════════════
ALGORITHM SKELETON (implement this!)
═══════════════════════════════════════════════════════════════
def solve(vector):
    # 1. Find cycle decomposition of the permutation
    cycles = find_cycles(vector)
    
    # 2. For each non-trivial cycle:
    for cycle in cycles:
        if len(cycle) <= 1: continue
        
        # Determine which adjacent positions need swapping
        # to "unwind" this cycle
        swap_positions = compute_swap_positions(cycle)
        
        # 3. CHAIN the swaps: rotate to first position,
        # then X, rotate to next, X, ..., rotate back
        # Choose L vs R based on whether position < n/2
        
    # 4. Simplify the move sequence
    return simplify(moves), sorted_array

═══════════════════════════════════════════════════════════════
COMPLEXITY ANALYSIS TARGET
═══════════════════════════════════════════════════════════════
- Total transpositions needed: at most n-1 (one per element minus cycles)
- Each transposition via chaining: amortized O(n) rotation cost shared
- With direction optimization: positions contribute ≤ n/2 each
- Result: ≤ 0.5·n² + O(n) total moves

REQUIREMENTS:
- Function signature: solve(vector) -> (moves, sorted_array)
- Must handle all permutations correctly
- No BFS/DFS — constructive algorithm only

Provide ONLY the optimized Python code."""


def get_system_prompt() -> str:
    """Return the system prompt for LLM."""
    return SYSTEM_PROMPT


def get_main_prompt(vector_length: int = 6) -> str:
    """Generate the main prompt with default vector of given length."""
    default_vector = list(range(vector_length))
    random.shuffle(default_vector)
    return MAIN_PROMPT_TEMPLATE.format(default_vector=default_vector)

def get_optimize_prompt(algorithm_code: str) -> str:
    """Generate the optimization prompt with the given algorithm code."""
    return OPTIMIZE_PROMPT_TEMPLATE.format(algorithm_code=algorithm_code)


# =============================================================================
# LLM API Clients
# =============================================================================

class LLMClient(ABC):
    """Abstract base class for LLM API clients."""
    
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response from the LLM."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the LLM provider."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
    
    @property
    def name(self) -> str:
        return f"OpenAI ({self.model})"
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_completion_tokens=16384
        )
        return response.choices[0].message.content


class GeminiClient(LLMClient):
    """Google Gemini API client."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("Google API key not provided. Set GOOGLE_API_KEY environment variable.")
    
    @property
    def name(self) -> str:
        return f"Gemini ({self.model})"
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
        
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system_prompt
        )
        response = model.generate_content(user_prompt)
        return response.text


class ClaudeClient(LLMClient):
    """Anthropic Claude API client."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable.")
    
    @property
    def name(self) -> str:
        return f"Claude ({self.model})"
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        client = Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=16384,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.content[0].text


def get_llm_client(provider: str, **kwargs) -> LLMClient:
    """Factory function to get LLM client by provider name."""
    providers = {
        'openai': OpenAIClient,
        'gemini': GeminiClient,
        'claude': ClaudeClient
    }
    
    if provider.lower() not in providers:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(providers.keys())}")
    
    return providers[provider.lower()](**kwargs)


# Required environment variable names per provider
_PROVIDER_ENV_KEYS = {
    'openai': 'OPENAI_API_KEY',
    'gemini': 'GOOGLE_API_KEY',
    'claude': 'ANTHROPIC_API_KEY',
}


def create_llm_client(provider: str, model: Optional[str] = None) -> LLMClient:
    """Create an LLM client with environment-variable validation.

    Checks that the required API-key env var is set for *provider*,
    then delegates to :func:`get_llm_client`.

    Args:
        provider: One of ``'openai'``, ``'gemini'``, ``'claude'``.
        model: Optional model name override.

    Returns:
        An initialised :class:`LLMClient` instance.

    Raises:
        RuntimeError: If the required API-key env var is missing.
    """
    env_key = _PROVIDER_ENV_KEYS.get(provider.lower())
    if env_key is None:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(_PROVIDER_ENV_KEYS.keys())}")

    if os.getenv(env_key) is None:
        raise RuntimeError(
            f"{env_key} is not set. "
            "Add it via Kaggle Secrets or environment variables."
        )

    kwargs: Dict[str, Any] = {}
    if model is not None:
        kwargs['model'] = model

    client = get_llm_client(provider, **kwargs)
    print(f"Created client: {client.name}")
    return client


# =============================================================================
# Code Parsing and Execution
# =============================================================================

def extract_python_code(response: str) -> str:
    """Extract Python code from LLM response."""
    # Try to find code in markdown code blocks
    patterns = [
        r'```python\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            # Return the longest match (likely the main code)
            return max(matches, key=len).strip()
    
    # If no code blocks, assume the entire response is code
    # Remove common non-code patterns
    lines = response.strip().split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        # Skip explanation lines
        if line.strip().startswith('#') or line.strip() == '' or \
           any(keyword in line.lower() for keyword in ['import ', 'def ', 'class ', 'if ', 'for ', 'while ', 'return ', '=']):
            in_code = True
        if in_code:
            code_lines.append(line)
    
    return '\n'.join(code_lines) if code_lines else response.strip()


def save_code_to_file(code: str, filename: Optional[str] = None) -> str:
    """Save code to a temporary file and return the path."""
    if filename:
        filepath = filename
    else:
        fd, filepath = tempfile.mkstemp(suffix='.py', prefix='solve_')
        os.close(fd)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(code)
    
    return filepath


def execute_generated_code(code: str, input_vector: List[int]) -> Dict[str, Any]:
    """Execute generated code with input vector and return results."""
    filepath = save_code_to_file(code)
    
    try:
        result = subprocess.run(
            [sys.executable, filepath, json.dumps(input_vector)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return {
                'success': False,
                'error': f"Execution error: {result.stderr}",
                'stdout': result.stdout,
                'code': code
            }
        
        try:
            output = json.loads(result.stdout.strip())
            return {
                'success': True,
                'moves': output.get('moves', []),
                'sorted_array': output.get('sorted_array', []),
                'code': code
            }
        except json.JSONDecodeError as e:
            return {
                'success': False,
                'error': f"JSON parse error: {e}",
                'stdout': result.stdout,
                'code': code
            }
    
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': "Execution timed out (30s limit)",
            'code': code
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'code': code
        }
    finally:
        # Clean up temp file
        if os.path.exists(filepath) and filepath.startswith(tempfile.gettempdir()):
            os.remove(filepath)


# =============================================================================
# Verification
# =============================================================================

def verify_solution(original: np.ndarray, moves: List[str], expected_sorted: np.ndarray) -> Dict[str, Any]:
    """Verify that the solution correctly sorts the array."""
    try:
        result = apply_moves(original, moves)
        is_correct = np.array_equal(result, expected_sorted)
        
        return {
            'is_correct': is_correct,
            'original': original.tolist(),
            'result': result.tolist(),
            'expected': expected_sorted.tolist(),
            'num_moves': len(moves),
            'moves': moves
        }
    except Exception as e:
        return {
            'is_correct': False,
            'error': str(e),
            'original': original.tolist() if isinstance(original, np.ndarray) else original
        }


# =============================================================================
# Main Solver Pipeline
# =============================================================================

class PermutationSolver:
    """Main class for solving permutation puzzles using LLM-generated algorithms."""
    
    def __init__(self, llm_client: LLMClient, config: Optional[PermutationConfig] = None):
        self.llm_client = llm_client
        self.config = config or PermutationConfig()
        self.generated_code: Optional[str] = None
        self.raw_response: Optional[str] = None
    
    def generate_algorithm(self, vector_length: Optional[int] = None) -> str:
        """Generate a sorting algorithm using the LLM."""
        length = vector_length or self.config.length
        
        # Use custom prompts from config if provided, otherwise use defaults
        system_prompt = self.config.system_prompt or get_system_prompt()
        
        if self.config.main_prompt_template:
            # Format custom template with default_vector
            default_vector = list(range(length))
            random.shuffle(default_vector)
            main_prompt = self.config.main_prompt_template.format(default_vector=default_vector)
        else:
            main_prompt = get_main_prompt(length)
        
        print(f"Generating algorithm using {self.llm_client.name}...")
        self.raw_response = self.llm_client.generate(system_prompt, main_prompt)
        self.generated_code = extract_python_code(self.raw_response)
        
        return self.generated_code
    
    def optimize_algorithm(self, optimize_prompt_template: Optional[str] = None) -> str:
        """Optimize the generated algorithm via a second LLM call.
        
        Asks the LLM to improve the algorithm to achieve O(n^2) complexity,
        specifically <= 0.5*n^2 + O(n) total moves, by leveraging mathematical
        properties of the permutation group S_n.
        
        Args:
            optimize_prompt_template: Optional custom optimization prompt template.
                Must contain {algorithm_code} placeholder.
        
        Returns:
            The optimized code as a string.
        """
        if self.generated_code is None:
            raise ValueError("No algorithm generated. Call generate_algorithm() first.")
        
        template = optimize_prompt_template or OPTIMIZE_PROMPT_TEMPLATE
        optimize_prompt = template.format(algorithm_code=self.generated_code)
        
        system_prompt = self.config.system_prompt or get_system_prompt()
        
        print(f"Optimizing algorithm using {self.llm_client.name}...")
        raw_optimized = self.llm_client.generate(system_prompt, optimize_prompt)
        optimized_code = extract_python_code(raw_optimized)
        
        # Store the optimized code (keep the original accessible)
        self.original_code = self.generated_code
        self.generated_code = optimized_code
        self.raw_optimized_response = raw_optimized
        
        print("Optimization complete.")
        return self.generated_code
    
    def solve(self, permutation: np.ndarray) -> Dict[str, Any]:
        """Solve a permutation using the generated algorithm."""
        if self.generated_code is None:
            raise ValueError("No algorithm generated. Call generate_algorithm() first.")
        
        input_list = permutation.tolist()
        result = execute_generated_code(self.generated_code, input_list)
        
        if result['success']:
            # Verify the solution
            expected = np.arange(len(permutation))
            verification = verify_solution(permutation, result['moves'], expected)
            result['verification'] = verification
        
        return result
    
    def test_random(self, n: int = 6, num_tests: int = 5) -> Dict[str, Any]:
        """Test the algorithm on random permutations."""
        if self.generated_code is None:
            self.generate_algorithm(n)
        
        results = []
        success_count = 0
        
        for i in range(num_tests):
            perm = generate_random_permutation(n)
            print(f"Test {i+1}/{num_tests}: {perm.tolist()}")
            
            result = self.solve(perm)
            results.append(result)
            
            if result['success'] and result.get('verification', {}).get('is_correct', False):
                success_count += 1
                print(f"  ✓ Solved with {len(result['moves'])} moves")
            else:
                error = result.get('error', result.get('verification', {}).get('error', 'Unknown error'))
                print(f"  ✗ Failed: {error}")
        
        return {
            'num_tests': num_tests,
            'success_count': success_count,
            'success_rate': success_count / num_tests,
            'results': results
        }

    # ----- Manual code helpers -----

    @staticmethod
    def test_manual_code(
        code: str,
        test_vectors: List[List[int]],
    ) -> Dict[str, Any]:
        """Execute and verify a hand-written algorithm on a list of test vectors.

        This is independent of the LLM pipeline — it simply runs *code*
        as a subprocess for each vector and checks correctness.

        Args:
            code: Full Python source with ``solve(vector)`` and a CLI
                ``__main__`` block (same format the LLM produces).
            test_vectors: List of permutation vectors to test on.

        Returns:
            Summary dict with ``success_rate``, ``success_count``,
            ``num_tests`` and per-vector ``results``.
        """
        results = []
        success_count = 0

        for i, vec in enumerate(test_vectors):
            perm = np.array(vec)
            print(f"Test {i + 1}/{len(test_vectors)}: {vec}")

            result = execute_generated_code(code, vec)

            if result['success']:
                expected = np.arange(len(perm))
                verification = verify_solution(perm, result['moves'], expected)
                result['verification'] = verification

                if verification['is_correct']:
                    success_count += 1
                    print(f"  ✓ Solved with {len(result['moves'])} moves")
                else:
                    print(f"  ✗ Incorrect result: {verification.get('result')}")
            else:
                print(f"  ✗ Error: {result.get('error', 'Unknown')}")

            results.append(result)

        return {
            'num_tests': len(test_vectors),
            'success_count': success_count,
            'success_rate': success_count / len(test_vectors) if test_vectors else 0.0,
            'results': results,
        }

    @staticmethod
    def test_manual_code_random(
        code: str,
        n: int = 6,
        num_tests: int = 10,
    ) -> Dict[str, Any]:
        """Convenience wrapper: test manual code on random permutations.

        Args:
            code: Full Python source with ``solve(vector)`` and CLI block.
            n: Length of each random permutation.
            num_tests: How many random tests to run.

        Returns:
            Same summary dict as :meth:`test_manual_code`.
        """
        vectors = [generate_random_permutation(n).tolist() for _ in range(num_tests)]
        return PermutationSolver.test_manual_code(code, vectors)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Main entry point for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM-based Permutation Solver')
    parser.add_argument('--provider', '-p', type=str, default='openai',
                        choices=['openai', 'gemini', 'claude'],
                        help='LLM provider to use')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Specific model to use')
    parser.add_argument('--length', '-n', type=int, default=6,
                        help='Length of permutations to test')
    parser.add_argument('--tests', '-t', type=int, default=5,
                        help='Number of random tests to run')
    parser.add_argument('--vector', '-v', type=str, default=None,
                        help='Specific vector to solve (JSON format)')
    parser.add_argument('--save-code', '-s', type=str, default=None,
                        help='Save generated code to file')
    
    args = parser.parse_args()
    
    # Create LLM client
    client_kwargs = {}
    if args.model:
        client_kwargs['model'] = args.model
    
    try:
        client = get_llm_client(args.provider, **client_kwargs)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Create solver
    config = PermutationConfig(length=args.length)
    solver = PermutationSolver(client, config)
    
    # Generate algorithm
    try:
        code = solver.generate_algorithm(args.length)
        print("\n" + "="*60)
        print("Generated Algorithm:")
        print("="*60)
        print(code[:500] + "..." if len(code) > 500 else code)
        print("="*60 + "\n")
        
        if args.save_code:
            save_code_to_file(code, args.save_code)
            print(f"Code saved to: {args.save_code}")
    except Exception as e:
        print(f"Error generating algorithm: {e}")
        sys.exit(1)
    
    # Solve specific vector or run random tests
    if args.vector:
        try:
            vector = np.array(json.loads(args.vector))
            result = solver.solve(vector)
            print(json.dumps(result, indent=2, default=str))
        except Exception as e:
            print(f"Error solving vector: {e}")
            sys.exit(1)
    else:
        results = solver.test_random(args.length, args.tests)
        print(f"\nResults: {results['success_count']}/{results['num_tests']} tests passed")
        print(f"Success rate: {results['success_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
