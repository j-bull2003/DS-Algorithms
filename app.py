from dash import Dash, html, dcc, Input, Output, State
import textwrap

app = Dash(__name__)

server = app.server

# Function to wrap code in HTML format
def format_code(code):
    return html.Pre(textwrap.dedent(code), style={'white-space': 'pre-wrap', 'font-family': 'monospace', 'background-color': '#f8f8f8', 'padding': '10px'})

# Define the layout of the app
app.layout = html.Div([
    html.H1(children='Data Structures and Algorithms Project'),

    # Dropdown to select data structure or algorithm
    dcc.Dropdown(
        id='topic-dropdown',
        options=[
            {'label': 'Bubble Sort', 'value': 'bubble_sort'},
            {'label': 'Selection Sort', 'value': 'selection_sort'},
            {'label': 'Insertion Sort', 'value': 'insertion_sort'},
            {'label': 'Shell Sort', 'value': 'shell_sort'},
            {'label': 'Radix Sort', 'value': 'radix_sort'},
            {'label': 'Merge Sort', 'value': 'merge_sort'},
            {'label': 'Quick Sort', 'value': 'quick_sort'},
            {'label': 'DFS Algorithm', 'value': 'dfs'},
            {'label': 'BFS Algorithm', 'value': 'bfs'}
        ],
        value='bubble_sort'  # Default value
    ),

    # Div to display content based on the selected topic
    html.Div(id='content-display'),

    # Input for algorithm interaction
    dcc.Input(id='input-data', type='text', placeholder='Enter data (comma separated)'),
    html.Button('Run Algorithm', id='run-button'),
    html.Div(id='output-display')
])

# Callback to update the content display based on the selected topic
@app.callback(
    Output('content-display', 'children'),
    [Input('topic-dropdown', 'value')]
)
def display_content(selected_topic):
    if selected_topic == 'bubble_sort':
        code_example = """
        # Bubble Sort implementation in Python
        def bubble_sort(arr):
            n = len(arr)
            for i in range(n):
                for j in range(0, n-i-1):
                    if arr[j] > arr[j+1]:
                        arr[j], arr[j+1] = arr[j+1], arr[j]

        # Example usage
        arr = [64, 34, 25, 12, 22, 11, 90]
        bubble_sort(arr)
        print("Sorted array is:", arr)  # Output: [11, 12, 22, 25, 34, 64, 90]
        """
        return html.Div([
            html.H2('Bubble Sort'),
            html.P('Bubble Sort is a simple sorting algorithm that repeatedly steps through the list, compares adjacent elements and swaps them if they are in the wrong order.'),
            html.Ul([
                html.Li('1. Scan the array left to right'),
                html.Li('2. Compare each adjacent elements'),
                html.Li('3. Swap them if they are out of order')
            ]),
            html.H3('Python Code Example:'),
            format_code(code_example)
        ])
    elif selected_topic == 'quick_sort':
        code_example = """
        # Quick Sort implementation in Python
        def partition(arr, low, high):
            i = (low-1)
            pivot = arr[high]

            for j in range(low, high):
                if arr[j] <= pivot:
                    i = i+1
                    arr[i], arr[j] = arr[j], arr[i]
            arr[i+1], arr[high] = arr[high], arr[i+1]
            return i+1

        def quick_sort(arr, low, high):
            if low < high):
                pi = partition(arr, low, high)
                quick_sort(arr, low, pi-1)
                quick_sort(arr, pi+1, high)

        # Example usage
        arr = [10, 7, 8, 9, 1, 5]
        n = len(arr)
        quick_sort(arr, 0, n-1)
        print("Sorted array is:", arr)  # Output: [1, 5, 7, 8, 9, 10]
        """
        return html.Div([
            html.H2('Quick Sort'),
            html.P('Quick Sort is an efficient sorting algorithm that uses a divide-and-conquer approach to sort elements by partitioning the array into smaller sub-arrays.'),
            html.Ul([
                html.Li('Efficient for large lists'),
                html.Li('Complex implementation')
            ]),
            html.H3('Python Code Example:'),
            format_code(code_example)
        ])
    elif selected_topic == 'insertion_sort':
        code_example = """
        # Insertion Sort implementation in Python
        def insertion_sort(arr):
            for i in range(1, len(arr)):
                key = arr[i]
                j = i - 1
                while j >= 0 and key < arr[j]:
                    arr[j + 1] = arr[j]
                    j -= 1
                arr[j + 1] = key

        # Example usage
        arr = [12, 11, 13, 5, 6]
        insertion_sort(arr)
        print("Sorted array is:", arr)  # Output: [5, 6, 11, 12, 13]
        """
        return html.Div([
            html.H2('Insertion Sort'),
            html.P('Insertion Sort is a simple sorting algorithm that builds the final sorted array one item at a time.'),
            html.Ul([
                html.Li('Efficient for small data sets'),
                html.Li('Simple to implement')
            ]),
            html.H3('Python Code Example:'),
            format_code(code_example)
        ])
    elif selected_topic == 'selection_sort':
        code_example = """
        # Selection Sort implementation in Python
        def selection_sort(arr):
            for i in range(len(arr)):
                min_idx = i
                for j in range(i+1, len(arr)):
                    if arr[j] < arr[min_idx]:
                        min_idx = j
                arr[i], arr[min_idx] = arr[min_idx], arr[i]

        # Example usage
        arr = [64, 25, 12, 22, 11]
        selection_sort(arr)
        print("Sorted array is:", arr)  # Output: [11, 12, 22, 25, 64]
        """
        return html.Div([
            html.H2('Selection Sort'),
            html.P('Selection Sort is a simple sorting algorithm that divides the array into a sorted and an unsorted region and repeatedly selects the smallest element from the unsorted region.'),
            html.Ul([
                html.Li('1. Find the minimum element in the unsorted array'),
                html.Li('2. Swap it with the first unsorted element'),
                html.Li('3. Move the boundary of the sorted region one step to the right')
            ]),
            html.H3('Python Code Example:'),
            format_code(code_example)
        ])
    elif selected_topic == 'shell_sort':
        code_example = """
        # Shell Sort implementation in Python
        def shell_sort(arr):
            n = len(arr)
            gap = n // 2
            while gap > 0:
                for i in range(gap, n):
                    temp = arr[i]
                    j = i
                    while j >= gap and arr[j - gap] > temp:
                        arr[j] = arr[j - gap]
                        j -= gap
                    arr[j] = temp
                gap //= 2

        # Example usage
        arr = [12, 34, 54, 2, 3]
        shell_sort(arr)
        print("Sorted array is:", arr)  # Output: [2, 3, 12, 34, 54]
        """
        return html.Div([
            html.H2('Shell Sort'),
            html.P('Shell Sort is an in-place comparison-based sorting algorithm. It is a generalization of insertion sort that allows the exchange of items that are far apart.'),
            html.Ul([
                html.Li('Uses a gap sequence to determine the elements to compare'),
                html.Li('Reduces the gap and sorts subarrays until the entire array is sorted')
            ]),
            html.H3('Python Code Example:'),
            format_code(code_example)
        ])
    elif selected_topic == 'radix_sort':
        code_example = """
        # Radix Sort implementation in Python
        def counting_sort(arr, exp1):
            n = len(arr)
            output = [0] * n 
            count = [0] * 10

            for i in range(0, n):
                index = arr[i] // exp1
                count[index % 10] += 1

            for i in range(1, 10):
                count[i] += count[i - 1]

            i = n - 1
            while i >= 0:
                index = arr[i] // exp1
                output[count[index % 10] - 1] = arr[i]
                count[index % 10] -= 1
                i -= 1

            for i in range(0, len(arr)):
                arr[i] = output[i]

        def radix_sort(arr):
            max1 = max(arr)
            exp = 1
            while max1 // exp > 0:
                counting_sort(arr, exp)
                exp *= 10

        # Example usage
        arr = [170, 45, 75, 90, 802, 24, 2, 66]
        radix_sort(arr)
        print("Sorted array is:", arr)  # Output: [2, 24, 45, 66, 75, 90, 170, 802]
        """
        return html.Div([
            html.H2('Radix Sort'),
            html.P('Radix Sort is a non-comparative sorting algorithm. It sorts numbers by processing individual digits.'),
            html.Ul([
                html.Li('Sorts based on the least significant digit to the most significant digit'),
                html.Li('Uses counting sort as a subroutine')
            ]),
            html.H3('Python Code Example:'),
            format_code(code_example)
        ])
    elif selected_topic == 'merge_sort':
        code_example = """
        # Merge Sort implementation in Python
        def merge_sort(arr):
            if len(arr) > 1:
                mid = len(arr) // 2
                L = arr[:mid]
                R = arr[mid:]

                merge_sort(L)
                merge_sort(R)

                i = j = k = 0
                while i < len(L) and j < len(R):
                    if L[i] < R[j]:
                        arr[k] = L[i]
                        i += 1
                    else:
                        arr[k] = R[j]
                        j += 1
                    k += 1

                while i < len(L):
                    arr[k] = L[i]
                    i += 1
                    k += 1

                while j < len(R):
                    arr[k] = R[j]
                    j += 1
                    k += 1

        # Example usage
        arr = [12, 11, 13, 5, 6, 7]
        merge_sort(arr)
        print("Sorted array is:", arr)  # Output: [5, 6, 7, 11, 12, 13]
        """
        return html.Div([
            html.H2('Merge Sort'),
            html.P('Merge Sort is a divide-and-conquer algorithm that splits the array into halves, recursively sorts them, and then merges the sorted halves.'),
            html.Ul([
                html.Li('Stable sort'),
                html.Li('Divides the array into halves and merges them after sorting each half')
            ]),
            html.H3('Python Code Example:'),
            format_code(code_example)
        ])
    elif selected_topic == 'dfs':
        code_example = """
        # DFS Algorithm implementation in Python
        def dfs(graph, start, visited=None):
            if visited is None:
                visited = set()
            visited.add(start)
            print(start, end=' ')

            for next in graph[start] - visited:
                dfs(graph, next, visited)
            return visited

        # Example usage
        graph = {
            'A': {'B', 'C'},
            'B': {'A', 'D', 'E'},
            'C': {'A', 'F'},
            'D': {'B'},
            'E': {'B', 'F'},
            'F': {'C', 'E'}
        }
        dfs(graph, 'A')  # Output: A B D E F C
        """
        return html.Div([
            html.H2('Depth-First Search (DFS)'),
            html.P('DFS is an algorithm for traversing or searching tree or graph data structures. The algorithm starts at the root node and explores as far as possible along each branch before backtracking.'),
            html.Ul([
                html.Li('Uses a stack to keep track of nodes to be visited'),
                html.Li('Can be implemented recursively or iteratively')
            ]),
            html.H3('Python Code Example:'),
            format_code(code_example)
        ])
    elif selected_topic == 'bfs':
        code_example = """
        # BFS Algorithm implementation in Python
        from collections import deque

        def bfs(graph, start):
            visited = set()
            queue = deque([start])
            visited.add(start)

            while queue:
                vertex = queue.popleft()
                print(vertex, end=' ')

                for neighbor in graph[vertex]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

        # Example usage
        graph = {
            'A': {'B', 'C'},
            'B': {'A', 'D', 'E'},
            'C': {'A', 'F'},
            'D': {'B'},
            'E': {'B', 'F'},
            'F': {'C', 'E'}
        }
        bfs(graph, 'A')  # Output: A B C D E F
        """
        return html.Div([
            html.H2('Breadth-First Search (BFS)'),
            html.P('BFS is an algorithm for traversing or searching tree or graph data structures. It starts at the tree root and explores all nodes at the present depth level before moving on to nodes at the next depth level.'),
            html.Ul([
                html.Li('Uses a queue to keep track of nodes to be visited'),
                html.Li('Explores nodes level by level')
            ]),
            html.H3('Python Code Example:'),
            format_code(code_example)
        ])
    else:
        return html.Div([
            html.P('Select a topic to learn about data structures and algorithms.')
        ])

# Callback to run the algorithm based on user input
@app.callback(
    Output('output-display', 'children'),
    [Input('run-button', 'n_clicks')],
    [State('topic-dropdown', 'value'), State('input-data', 'value')]
)
def run_algorithm(n_clicks, selected_topic, input_data):
    if not n_clicks:
        return ''

    if not input_data:
        return 'Please enter some data to run the algorithm.'

    try:
        data = list(map(int, input_data.split(',')))
    except ValueError:
        return 'Invalid input data. Please enter a comma-separated list of numbers.'

    if selected_topic == 'bubble_sort':
        def bubble_sort(arr):
            n = len(arr)
            for i in range(n):
                for j in range(0, n-i-1):
                    if arr[j] > arr[j+1]:
                        arr[j], arr[j+1] = arr[j+1], arr[j]
            return arr

        sorted_data = bubble_sort(data)
        return f'Sorted array: {sorted_data}'

    elif selected_topic == 'quick_sort':
        def partition(arr, low, high):
            i = (low-1)
            pivot = arr[high]

            for j in range(low, high):
                if arr[j] <= pivot:
                    i = i+1
                    arr[i], arr[j] = arr[j], arr[i]
            arr[i+1], arr[high] = arr[high], arr[i+1]
            return i+1

        def quick_sort(arr, low, high):
            if low < high:
                pi = partition(arr, low, high)
                quick_sort(arr, low, pi-1)
                quick_sort(arr, pi+1, high)
            return arr

        sorted_data = quick_sort(data, 0, len(data)-1)
        return f'Sorted array: {sorted_data}'

    elif selected_topic == 'insertion_sort':
        def insertion_sort(arr):
            for i in range(1, len(arr)):
                key = arr[i]
                j = i - 1
                while j >= 0 and key < arr[j]:
                    arr[j + 1] = arr[j]
                    j -= 1
                arr[j + 1] = key
            return arr

        sorted_data = insertion_sort(data)
        return f'Sorted array: {sorted_data}'

    elif selected_topic == 'selection_sort':
        def selection_sort(arr):
            for i in range(len(arr)):
                min_idx = i
                for j in range(i+1, len(arr)):
                    if arr[j] < arr[min_idx]:
                        min_idx = j
                arr[i], arr[min_idx] = arr[min_idx], arr[i]
            return arr

        sorted_data = selection_sort(data)
        return f'Sorted array: {sorted_data}'

    elif selected_topic == 'shell_sort':
        def shell_sort(arr):
            n = len(arr)
            gap = n // 2
            while gap > 0:
                for i in range(gap, n):
                    temp = arr[i]
                    j = i
                    while j >= gap and arr[j - gap] > temp:
                        arr[j] = arr[j - gap]
                        j -= gap
                    arr[j] = temp
                gap //= 2
            return arr

        sorted_data = shell_sort(data)
        return f'Sorted array: {sorted_data}'

    elif selected_topic == 'radix_sort':
        def counting_sort(arr, exp1):
            n = len(arr)
            output = [0] * n 
            count = [0] * 10

            for i in range(0, n):
                index = arr[i] // exp1
                count[index % 10] += 1

            for i in range(1, 10):
                count[i] += count[i - 1]

            i = n - 1
            while i >= 0:
                index = arr[i] // exp1
                output[count[index % 10] - 1] = arr[i]
                count[index % 10] -= 1
                i -= 1

            for i in range(0, len(arr)):
                arr[i] = output[i]

        def radix_sort(arr):
            max1 = max(arr)
            exp = 1
            while max1 // exp > 0:
                counting_sort(arr, exp)
                exp *= 10

        sorted_data = radix_sort(data)
        return f'Sorted array: {sorted_data}'

    elif selected_topic == 'merge_sort':
        def merge_sort(arr):
            if len(arr) > 1:
                mid = len(arr) // 2
                L = arr[:mid]
                R = arr[mid:]

                merge_sort(L)
                merge_sort(R)

                i = j = k = 0
                while i < len(L) and j < len(R):
                    if L[i] < R[j]:
                        arr[k] = L[i]
                        i += 1
                    else:
                        arr[k] = R[j]
                        j += 1
                    k += 1

                while i < len(L):
                    arr[k] = L[i]
                    i += 1
                    k += 1

                while j < len(R):
                    arr[k] = R[j]
                    j += 1
                    k += 1
            return arr

        sorted_data = merge_sort(data)
        return f'Sorted array: {sorted_data}'

    elif selected_topic == 'dfs':
        graph = {
            'A': {'B', 'C'},
            'B': {'A', 'D', 'E'},
            'C': {'A', 'F'},
            'D': {'B'},
            'E': {'B', 'F'},
            'F': {'C', 'E'}
        }
        
        def dfs(graph, start, visited=None):
            if visited is None:
                visited = set()
            visited.add(start)
            result = [start]

            for next in graph[start] - visited:
                result.extend(dfs(graph, next, visited))
            return result
        
        try:
            sorted_data = dfs(graph, data[0])
            return f'Visited nodes in order: {sorted_data}'
        except KeyError:
            return 'Invalid starting node for DFS. Please enter a valid node.'

    elif selected_topic == 'bfs':
        from collections import deque
        graph = {
            'A': {'B', 'C'},
            'B': {'A', 'D', 'E'},
            'C': {'A', 'F'},
            'D': {'B'},
            'E': {'B', 'F'},
            'F': {'C', 'E'}
        }

        def bfs(graph, start):
            visited = set()
            queue = deque([start])
            visited.add(start)
            result = []

            while queue:
                vertex = queue.popleft()
                result.append(vertex)

                for neighbor in graph[vertex]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            return result
        
        try:
            sorted_data = bfs(graph, data[0])
            return f'Visited nodes in order: {sorted_data}'
        except KeyError:
            return 'Invalid starting node for BFS. Please enter a valid node.'

    return 'Algorithm run successfully.'


if __name__ == '__main__':
    app.run_server(debug=True)
