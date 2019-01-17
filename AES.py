'''
Neal Bayya, January 2019
Implementation of Advanced Encryption Standard - 128 bit version
'''
import numpy as np
'''
DECRYPTION CODE: The following section of code includes inverse AES operations
Breakdown of decryption code structure
    1) Inverted Byte Sub (IBS)
        1a) ISBOX
    2) Inverted Shift Row (ISR)
    3) Inverted Mix Columns (IMC)
        3a) IMC_dot
    4) Inverted AddRoundKey (IARK)
    5) decode
    6) format_cipher_vec & get_decoded_text
'''

def ISBOX(element):
    '''
    Returns the input (single byte) of S-box needed to obtain 'element'
    '''
    ISBOX_lookup = [[0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB],
    [0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB],
    [0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E],
    [0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25],
    [0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92],
    [0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84],
    [0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06],
    [0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B],
    [0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73],
    [0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E],
    [0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B],
    [0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4],
    [0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F],
    [0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF],
    [0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61],
    [0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D]] #inverse mapping
    #rows represent left hex digit, columns represent right hex digit
    hex0 = element % 16
    hex1 = element // 16
    return ISBOX_lookup[hex1][hex0]

def IBS(inmat):
    '''
    Vectorize the functionality of ISBOX to work on matrices
    '''
    inv_vsbox = np.vectorize(ISBOX)
    return inv_vsbox(inmat)

def ISR(inmat):
    '''
    Rows of inmat are shifted to the right by 0,1,2,3 respectively (with wrapping)
    '''
    row1 = np.roll(inmat[1], 1) #np.roll shifts elements in row vector to the right
    row2 = np.roll(inmat[2], 2)
    row3 = np.roll(inmat[3], 3)
    outmat = np.copy(inmat)
    outmat[1] = row1
    outmat[2] = row2
    outmat[3] = row3
    return outmat

def IMC_dot(coeffs, v):
    '''
    Lower level helper method for IMC: Computes the dot product of a coefficients vector
    and input under multiplucation and addition constraints of AES (must keep elements to 1 byte)
    '''
    acc = 0 #XOR sum of products of coefficients and input vals
    for coeff, e in zip(coeffs, v):
        #Long multiplucation tableau of coeff and e
        r1bit_mult = e if (coeff & 1) else 0 #Right-most bit-string
        r2bit_mult = e << 1 if (coeff & 2) else 0
        r3bit_mult = e << 2 if (coeff & 4) else 0
        r4bit_mult = e << 3 if (coeff & 8) else 0 #Left-most bit-string
        xor_mult = r1bit_mult ^ r2bit_mult ^ r3bit_mult ^ r4bit_mult
        #Corrections for bit-string > 1 byte
        if xor_mult  >= (1<<10): #Force into 10 bits
            xor_mult ^= 0b10001101100
        if xor_mult  >= (1<<9): #Force into 9 bits
            xor_mult ^= 0b1000110110
        if xor_mult  >= (1<<8): #Force into 8 bits
            xor_mult ^= 0b100011011
        acc ^= xor_mult
    return acc

def IMC(inmat):
    '''
    High level code for matrix multiplucation to invert the mixed columns function
    '''
    MI = np.array([[14, 11, 13, 9],
    [9, 14, 11, 13],
    [13, 9, 14, 11],
    [11, 13, 9, 14]], dtype=int)
    outmat = np.zeros((4,4), dtype=int)
    for i in range(4):
        MI_row = MI[i] #row of coefficients
        outmat_row = np.zeros((4,), dtype=int)
        for j in range(4):
            C_col = inmat[:,j] #column of inmat
            outmat_row[j] = IMC_dot(MI_row, C_col) #perform sum of products under AES constraints
        outmat[i] = outmat_row #populate the output matrix
    return outmat

def IARK(K, A):
    '''
    Forward AES progresses uses ARK in a way that "adds" key matrix to output of mixed colums
    This function reverses AES in such context to obtain the input encrypted up to and including shift row
    Note: Function only to be performaed after IMC
    '''
    return np.bitwise_xor(IMC(K), A) #Calculate mixed columns of key and "add" to function input

def decode(K0, A10):
    '''
    Inputs: K0 is the initial 4x4 key used during encryption. A10 is the array of 4x4 matrices encrypted
    with 10 rounds of AES
    Outputs: A vector of the decoded message represented as numbers
    Notes: Function outlines the reversal of AES encode; numbers after variable names
        refer to the number of rounds the variable is still encrypted with
    '''
    K_rounds = generate_key_schedule(K0)
    nblocks = A10.shape[0]
    decoded_vec = np.zeros((16*nblocks,), dtype=int) #output vector to be populated
    for b in range(nblocks): #loop through all blocks of 4x4 encypted matrices
        #ROUND 0
        A10b = A10[b,:,:] #encryption block at index b
        K10 = K_rounds[10,:,:]
        A_prev = ARK(K10, A10b)
        #ROUNDS 1-9
        for round in range(1,10):
            Br = IBS(A_prev)
            Cr = ISR(Br)
            Dr = IMC(Cr)
            A_prev = IARK(K_rounds[10-round,:,:], Dr) #IARK links each round to the next
        #ROUND 10
        B0 = IBS(A_prev)
        C0 = ISR(B0)
        decoded_block = ARK(K_rounds[0,:,:], C0) # = A0
        decoded_vec[16*b:16*(b+1)] = decoded_block.flatten('F') #flatten A0 into vector
    return decoded_vec

def get_decoded_text(decrypted_vec):
    '''
    Converts output vector from decode() into text
    '''
    return "".join([chr(d) for d in decrypted_vec])

def format_cipher_vec(cipher_vec):
    '''
    Input: Vector of numbers that which are encoded with AES. Length is expected
    to be a multiple of 16 (to separate into 4x4 blocks)
    Output: 3-D matrix of all 4x4 cipher blocks, populated column-major
    '''
    nblocks = cipher_vec.shape[0] // 16
    A10 = np.zeros((nblocks,4,4), dtype=int) #A10 is encrypted with 10 rounds of AES
    for i in range(nblocks):
        cipher_block = np.reshape(cipher_vec[16*i: 16*(i+1)], (4,4))
        cipher_block = np.transpose(cipher_block) #vector must be read column wise
        A10[i,:,:] = cipher_block #populate output
    return A10

'''
ENCRYPTION CODE: The following section of code includes forward AES operations
Breakdown of encyption code structure
    1) Byte Sub (BS)
        1a) SBOX
    2) Shift Row (SR)
    3) Mix Columns (MC)
        3a) MC_dot
    4) AddRoundKey (ARK)
    5) transform_keycol
    6) encode
    6) format_plaintext & format_key
'''

def SBOX(element):
    '''
    Encrypts an element (1 byte) based on its two hexadecimal digits. The encryption is 1 to 1
    '''
    SBOX_lookup = [[0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76],
    [0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0],
    [0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15],
    [0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75],
    [0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84],
    [0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF],
    [0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8],
    [0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2],
    [0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73],
    [0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB],
    [0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79],
    [0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08],
    [0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A],
    [0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E],
    [0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF],
    [0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16]] #Lookup table for SBOX
    #rows represent left digit, columns represent right digit
    hex0 = element % 16
    hex1 = element // 16
    return SBOX_lookup[hex1][hex0]

def BS(inmat):
    '''
    Vectorizes the functionality of SBOX to work on matrices element wise
    '''
    vsbox = np.vectorize(SBOX)
    return vsbox(inmat)

def SR(inmat):
    '''
    Rows of inmat are shifted to the left by 0,1,2,3 respectively (with wrapping)
    '''
    row1 = np.roll(inmat[1], 3)
    row2 = np.roll(inmat[2], 2)
    row3 = np.roll(inmat[3], 1)
    outmat = np.copy(inmat)
    outmat[1] = row1
    outmat[2] = row2
    outmat[3] = row3
    return outmat

def MC_dot(coeffs, v):
    '''
    Lower level helper method for MC: Computes the dot product of a coefficients vector
    and input under multiplucation and addition constraints of AES (must keep elements to 1 byte)
    Coefficients are expected to be 1,2,or 3
    '''
    acc = 0 #Sum of products to be reported
    for coeff, e in zip(coeffs,v): #loop through corresponding input values and coefficients
        if coeff == 1: #coefficient of 1 does not change input
            acc ^= e
            continue
        #Long multiplucation for coeffs 2 & 3
        rbit_mult = e if coeff == 3 else 0 #only 3 has right bit string
        lbit_mult = e << 1
        lr_xor = rbit_mult ^ lbit_mult
        if lr_xor >> 8 == 1: #Force into 8 bits
            lr_xor ^= 0b100011011
        acc ^= lr_xor
    return acc

def MC(inmat):
    '''
    High level code for matrix multiplucation in mixed columns
    '''
    M = np.array([[2,3,1,1],
        [1,2,3,1],
        [1,1,2,3],
        [3,1,1,2]], dtype=int) #Coefficients for AES MC encryption
    outmat = np.zeros((4,4), dtype=int)
    for i in range(4):
        M_row = M[i] #Row of coefficients for mat mult
        outmat_row = np.zeros((4,), dtype=int)
        for j in range(4):
            C_col = inmat[:,j] #Col of input for mat mult
            outmat_row[j] = MC_dot(M_row, C_col) #Sum of products under AES byte constraints
        outmat[i] = outmat_row
    return outmat

def ARK(K, P):
    '''
    Input: 4x4 key matrix and 4x4 input matrix
    Output: 4x4 matrix that convolutes both matrices with xor
    Note: This function is used during both encoding and decoding because
        xor is the inverse of itself
    '''
    return np.bitwise_xor(K, P)

def transform_keycol(k_col, last_col, round_constant):
    '''
    Description: helper to generate_key_schedule method
    Inputs: k_col is the column of the key matrix to be trainsformed,
    last_col is a boolean value indicating whether the previous column was last in the key block,
    round_constant is used to transform the column but depends on which key block is being generated
    Ouput: Transformed column vector
    '''
    if not last_col: #Only last columns get transformed
        return k_col
    k_col = np.roll(k_col, 3) #Shift elements up 1 space with wrapping
    k_col = BS(k_col) #Apply S-Box on all elements
    k_col_entry0 = k_col[0] ^ round_constant #First element convoluted with round constant
    k_col[0] = k_col_entry0
    return k_col

def encode(K0, P):
    '''
    High Level method to encode a plaintext using AES operations
    Input: K0 is the initial 4x4 key matrix, P is the 3-D array of 4x4 plaintext blocks
    Output: vector of numbers that encrypt P using 10 rounds of AES
    '''
    K_rounds = generate_key_schedule(K0) #Key matrices for 0 round + all 10 rounds
    nblocks = P.shape[0] #Number of 4x4 blocks user is encoding
    cipher_vec = np.zeros((16*nblocks,), dtype=int)
    for b in range(nblocks): #loop through all plaintext blocks
        #ROUND 0
        Pb = P[b,:,:] #Plaintext block at index b
        K0 = K_rounds[0,:,:]
        A_prev = ARK(K0, Pb)
        #ROUNDS 1-9
        for round in range(9):
            Br = BS(A_prev)
            Cr = SR(Br)
            Dr = MC(Cr)
            A_prev = ARK(K_rounds[round+1,:,:], Dr) #ARK links each round to the next
        #ROUND 10
        B10 = BS(A_prev)
        C10 = SR(B10)
        cipher_block = ARK(K_rounds[10,:,:], C10) # = A10
        cipher_vec[16*b:16*(b+1)] = cipher_block.flatten('F') #flatten cipher matrix into vector and populate output
    return cipher_vec

def generate_key_schedule(K0):
    '''
    Generate key matrices for all 10 rounds of encryption/decryption of AES give initial key matrix
    Output is a matrix of 11 4x4 key matrices (K0 included in output)
    '''
    rounds = [0b00000001, 0b00000010, 0b00000100, 0b00001000, 0b00010000,
    0b00100000, 0b01000000, 0b10000000, 0b00011011, 0b00110110] #Round constants used in transformations
    prev_key_matrix = K0 #Keeps track of previous key block
    prev_col = K0[:,3] #Keeps track of last column in memory for transformations
    K_rounds = [K0] #Stores all rounds
    for round in range(10):
        Kr = [] #Stores Key matrix for given round
        for i in range(4):
            prev_key_i = prev_key_matrix[:,i] #Corresponding column in previous key block
            Kr_i = np.bitwise_xor(prev_key_i, transform_keycol(prev_col, i==0, rounds[round])) #Generate using transformation
            prev_col = Kr_i
            Kr.append(Kr_i)
        Kr = np.column_stack(Kr) #Vectors are generated column wise
        K_rounds.append(Kr)
        prev_key_matrix = Kr
    return np.stack(K_rounds)

def format_plaintext(plaintext):
    '''
    Takes in text input, pads text, and stores in numerical matrix of 4x4 blocks
    '''
    if len(plaintext) % 16 > 0: #Add Padding
        pad_amt = 16 - (len(plaintext)%16)
        pad = ''.join([' ' for i in range(pad_amt)])
        plaintext += pad
    block_matrices = list()
    for i in range(0,len(plaintext),16):
        #Convert every 16 characters into matrix
        block_list = np.array([ord(c) for c in plaintext[i:i+16]], dtype=int) #characters -> numbers
        block_matrix = np.reshape(block_list, (4,4))
        block_matrix = np.transpose(block_matrix) #Matrix generated column wise
        block_matrices.append(block_matrix)
    return np.stack(block_matrices)

def format_key(keytext):
    '''
    Takes in text input for key, pads/prunes, and stores in numerical 4x4 matrix
    '''
    #Ensure keytext is of length 16
    if len(keytext) < 16: #Padding
        pad = ''.join([' ' for i in range(16-len(keytext))])
        keytext += pad
    elif len(keytext) > 16: #Pruning
        keytext = keytext[:16]
    key_nums = np.array([ord(c) for c in keytext], dtype=int) #characters -> numbers
    key_matrix = np.reshape(key_nums, (4,4))
    return np.transpose(key_matrix) #Characters read into matrix column wise

def main():
    #Debugging tools to see numbers in binary and hex
    display_hex = np.vectorize(lambda n: format(n, 'x'))
    display_bin = np.vectorize(lambda n: format(n, 'b'))

    #Encrypting Phase
    plaintext = 'Hi my name is Neal Bayya and this text must be more than 16 characters'
    keyword = 'Neal Ratan Bayya'
    P = format_plaintext(plaintext)
    K0 = format_key(keyword)

    print("keyword: {}".format(K0))
    print("plaintext: {}".format(P))

    cipher_vec = encode(K0, P)
    print("cipher vector: {}".format(display_hex(cipher_vec)))

    #Decrypting Phase
    A10 = format_cipher_vec(cipher_vec)
    print("formatted A10: {}".format(display_hex(A10)))
    decoded = decode(K0, A10)
    print("decoded: {}".format(decoded))
    decoded_text = get_decoded_text(decoded)
    print("decoded text: {}".format(decoded_text))

if __name__ == '__main__':
    main()
