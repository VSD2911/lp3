#HUFFMAN ENCODING
import heapq
from collections import Counter, namedtuple

# node created with square bracket
Node=namedtuple("Node",["char","frequency","left","right"])


def build_huffman_tree(data):
    frequency=Counter(data)
# list of nodes with square and object of Node with round bracket
    nodes=[Node(char,freq,None,None)for char,freq in frequency.items()]
    
    while len(nodes) > 1:
        nodes.sort(key=lambda x:x.frequency)
        left=nodes.pop(0)
        right=nodes.pop(0)
#         object of Node wit round bracket
        parent=Node(None,left.frequency+right.frequency,left,right)
        nodes.append(parent)
        
    return nodes[0]

def build_huffman_codes(node,current_code="",huffman_codes={}):
    if node.char is not None:
        huffman_codes[node.char]=current_code
    if node.left:
        build_huffman_codes(node.left,current_code+ "0",huffman_codes)
    if node.right:
        build_huffman_codes(node.right,current_code+ "1",huffman_codes)
        
def huffman_encoding(data):
    if not data:
        return None, None
    root=build_huffman_tree(data)
    huffman_codes={}
    build_huffman_codes(root, "",huffman_codes)
    
    encoded_data="".join(huffman_codes[char]for char in data)
    
    return encoded_data,root


def huffman_decoding(encoded_data,root):
    if encoded_data is None or root is None:
        return None
    decoded_data=""
    current_node=root
    for bit in encoded_data:
        if bit=='0':
            current_node=current_node.left
        else:
            current_node=current_node.right
        if current_node.char is not None:
#             character has to be appended
            decoded_data+=current_node.char
            current_node=root
    return decoded_data

data="abracadabra"
encoded_data,tree=huffman_encoding(data)
print("Encoded data=",encoded_data)
# two arg encoded data and root node 
decoded_data=huffman_decoding(encoded_data,tree)
print("Decoded data= ",decoded_data)



# Time Complexity Analysis:
# - Building the Huffman tree: O(n*log(n)), where n is the number of unique characters.
# - Encoding data: O(m), where m is the length of the input data.
# - Decoding data: O(m), where m is the length of the encoded data.
# Overall time complexity: O(n*log(n)) + O(m) + O(m), which simplifies to O(n*log(n) + m).

# Space Complexity Analysis:
# - Storing character frequencies: O(n), where n is the number of unique characters.
# - Priority queue (min heap): O(n) to store nodes.
# - Encoding map: O(n) to store Huffman codes.
# - Encoded data: O(m), where m is the length of the input data.
# Overall space complexity: O(n + n + n + m), which simplifies to O(n + m).

"""
**Huffman Coding** is a widely used data compression algorithm that is capable of encoding and decoding data in a lossless manner. It was developed by David A. Huffman in 1952. The main idea behind Huffman coding is to assign shorter codes to more frequent characters and longer codes to less frequent characters, reducing the average length of the encoded data. Huffman coding is commonly used in file compression formats like ZIP, as well as in data transmission protocols.

Here's a step-by-step explanation of how Huffman coding works for encoding and decoding:

**Encoding:**
1. **Frequency Analysis:** In the encoding process, the first step is to analyze the input data to determine the frequency of each character (or symbol) that needs to be encoded. The more frequent a character, the shorter its Huffman code will be.

2. **Build Huffman Tree:** Next, a Huffman tree (also known as a Huffman encoding tree) is constructed. This is a binary tree where each leaf node represents a character and its frequency, and internal nodes represent the sum of the frequencies of their children. The tree is built by repeatedly merging the two nodes with the lowest frequencies, creating a new internal node as their parent, until there is only one node left, which becomes the root of the tree.

3. **Assign Codes:** To create Huffman codes, you traverse the Huffman tree from the root to each leaf node. When you move left in the tree, you append a '0' to the code, and when you move right, you append a '1'. The resulting codes for each character are unique and variable in length.

4. **Encode Data:** Finally, you encode the input data by replacing each character with its corresponding Huffman code. The encoded data is typically a binary string, where the Huffman codes are concatenated to represent the original data.

**Decoding:**
1. **Huffman Tree:** To decode the data, you need the same Huffman tree that was used for encoding. It's crucial to have the original tree structure to map the Huffman codes back to their respective characters.

2. **Decode Data:** Starting at the root of the Huffman tree, you read the encoded data bit by bit. When you encounter a '0', you move left in the tree, and when you encounter a '1', you move right. You continue this process until you reach a leaf node, which represents a character. That character is the decoded output. You then start over at the root to decode the next character in the input data.

**Advantages of Huffman Coding:**
- Huffman codes are variable-length and prefix-free, meaning no code is the prefix of another, making decoding unambiguous.
- It provides an efficient way to compress data, particularly when the input has variable symbol frequencies.

**Limitations of Huffman Coding:**
- Huffman encoding requires transmitting the tree structure alongside the encoded data, which can add overhead.
- It may not be the most efficient compression method for all types of data. Other algorithms, like Lempel-Ziv-Welch (LZW), are better for certain data patterns.

Huffman coding is a fundamental concept in data compression and forms the basis for many modern compression techniques. It's widely used in applications where efficient data storage and transmission are essential, such as in file compression, image compression, and data communication.

------------------------------------------------

Fixed-Length Encoding and Variable-Length Encoding are two different methods used in data compression and representation. Here are the key differences between them:

1. Length of Encoded Units:

Fixed-Length Encoding: In fixed-length encoding, each unit of data (such as a character or symbol) is represented using a fixed number of bits. Regardless of the frequency or importance of a character, it is encoded using the same number of bits.

Variable-Length Encoding: In variable-length encoding, the length of the code for each unit of data varies. More frequent or important characters are typically represented using shorter codes, while less frequent characters use longer codes. This results in a more efficient representation for data with non-uniform character frequencies.

2. Compression Efficiency:

Fixed-Length Encoding: Fixed-length encoding is not very efficient for compressing data. It uses the same amount of space for each unit, which can be wasteful for characters that occur infrequently.

Variable-Length Encoding: Variable-length encoding is more efficient for compressing data, especially when there is non-uniformity in character frequencies. It allows more frequent characters to be represented using shorter codes, which reduces the overall size of the encoded data.

3. Decoding Complexity:

Fixed-Length Encoding: Decoding fixed-length encoding is straightforward and simple. You can directly map the fixed-length codes to their corresponding characters.

Variable-Length Encoding: Decoding variable-length encoding requires more complex algorithms. Decoders must determine the boundaries between codes and map variable-length codes to their corresponding characters. Huffman coding is an example of a variable-length encoding method.

4. Space Overhead:

Fixed-Length Encoding: Fixed-length encoding typically results in some space overhead because shorter codes are used for less frequent characters. However, this overhead is often manageable.

Variable-Length Encoding: Variable-length encoding can be more space-efficient, especially for text data, as it assigns shorter codes to more frequent characters, reducing the overall size.

5. Examples:

Fixed-Length Encoding: An example of fixed-length encoding is ASCII (American Standard Code for Information Interchange), where each character is represented using 7 or 8 bits.

Variable-Length Encoding: Huffman coding and Lempel-Ziv-Welch (LZW) compression are examples of variable-length encoding used in data compression algorithms.

6. Common Use Cases:

Fixed-Length Encoding: Fixed-length encoding is used in situations where simplicity and constant-time access are more important than space efficiency. For example, it may be used in certain hardware protocols or file formats.

Variable-Length Encoding: Variable-length encoding is commonly used in data compression, including text compression (e.g., ZIP files), image compression (e.g., JPEG), and video compression (e.g., H.264). It is also used in data transmission protocols where efficiency is a priority.

In summary, fixed-length encoding assigns a fixed number of bits to each character, while variable-length encoding uses shorter codes for more frequent characters and longer codes for less frequent ones. Variable-length encoding is more space-efficient and is commonly used in data compression.
