#HUFFMAN ENCODING
from collections import Counter, namedtuple
import heapq

Node = namedtuple("Node", ["char", "frequency", "left", "right"])

def build_huffman_tree(data):
    # Count the frequency of each character in the data
    frequency = Counter(data)

    # Create leaf nodes for each character
    nodes = [Node(char, freq, None, None) for char, freq in frequency.items()]

    # Build the Huffman tree
    while len(nodes) > 1:
        # Sort nodes by frequency
        nodes.sort(key=lambda x: x.frequency)
        left = nodes.pop(0)
        right = nodes.pop(0)
        # Create a parent node with the sum of frequencies
        parent = Node(None, left.frequency + right.frequency, left, right)
        nodes.append(parent)

    return nodes[0]  # The root of the Huffman tree

def build_huffman_codes(node, current_code="", huffman_codes={}):
    if node.char is not None:
        huffman_codes[node.char] = current_code
    if node.left:
        build_huffman_codes(node.left, current_code + "0", huffman_codes)
    if node.right:
        build_huffman_codes(node.right, current_code + "1", huffman_codes)

def huffman_encoding(data):
    if not data:
        return None, None

    root = build_huffman_tree(data)
    huffman_codes = {}
    build_huffman_codes(root, "", huffman_codes)

    encoded_data = "".join(huffman_codes[char] for char in data)
    return encoded_data, root

def huffman_decoding(encoded_data, root):
    if encoded_data is None or root is None:
        return None

    decoded_data = ""
    current_node = root

    for bit in encoded_data:
        if bit == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right

        if current_node.char is not None:
            decoded_data += current_node.char
            current_node = root

    return decoded_data

# Example usage
if __name__ == "__main__":
    data = "this is an example for huffman encoding"
    encoded_data, tree = huffman_encoding(data)
    print(f"Encoded data: {encoded_data}")
    
    decoded_data = huffman_decoding(encoded_data, tree)
    print(f"Decoded data: {decoded_data}")