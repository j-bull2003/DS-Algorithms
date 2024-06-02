from dash import Dash, html, dcc, Input, Output, State
import textwrap

# Initialize the Dash app
app = Dash(__name__)

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
            {'label': 'Binary Tree', 'value': 'binary_tree'},
            {'label': 'Bubble Sort', 'value': 'bubble_sort'},
            {'label': 'Quick Sort', 'value': 'quick_sort'},
            {'label': 'Insertion Sort', 'value': 'insertion_sort'},
            {'label': 'Binary Sort', 'value': 'binary_sort'}
        ],
        value='array'  # Default value
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
                html.Li('Simple to understand and implement'),
                html.Li('Inefficient for large lists')
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
            if low < high:
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
    elif selected_topic == 'binary_sort':
        code_example = """
        # Binary Insertion Sort implementation in Python
        def binary_search(arr, val, start, end):
            if start == end:
                if arr[start] > val:
                    return start
                else:
                    return start + 1
            if start > end:
                return start

            mid = (start + end) // 2
            if arr[mid] < val:
                return binary_search(arr, val, mid + 1, end)
            elif arr[mid] > val:
                return binary_search(arr, val, start, mid - 1)
            else:
                return mid

        def binary_insertion_sort(arr):
            for i in range(1, len(arr)):
                val = arr[i]
                j = binary_search(arr, val, 0, i - 1)
                arr = arr[:j] + [val] + arr[j:i] + arr[i + 1:]
            return arr

        # Example usage
        arr = [37, 23, 0, 17, 12, 72, 31, 46, 100, 88, 54]
        sorted_arr = binary_insertion_sort(arr)
        print("Sorted array is:", sorted_arr)  # Output: [0, 12, 17, 23, 31, 37, 46, 54, 72, 88, 100]
        """
        return html.Div([
            html.H2('Binary Insertion Sort'),
            html.P('Binary Insertion Sort uses binary search to find the correct location to insert the selected item at each iteration.'),
            html.Ul([
                html.Li('More efficient than simple insertion sort for large arrays')
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

    elif selected_topic == 'binary_sort':
        def binary_search(arr, val, start, end):
            if start == end:
                if arr[start] > val:
                    return start
                else:
                    return start + 1
            if start > end:
                return start

            mid = (start + end) // 2
            if arr[mid] < val:
                return binary_search(arr, val, mid + 1, end)
            elif arr[mid] > val:
                return binary_search(arr, val, start, mid - 1)
            else:
                return mid

        def binary_insertion_sort(arr):
            for i in range(1, len(arr)):
                val = arr[i]
                j = binary_search(arr, val, 0, i - 1)
                arr = arr[:j] + [val] + arr[j:i] + arr[i + 1:]
            return arr

        sorted_data = binary_insertion_sort(data)
        return f'Sorted array: {sorted_data}'

    return 'Algorithm run successfully.'

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
