# Security Analysis and Methodology: SHA-512 Stream Cipher

## 1. Methodology

The implementation transforms the standard SHA-512 hash function (which normally produces a fixed 64-byte output) into a **Stream Cipher** to encrypt an image of any size.

### The Process
1.  **Keystream Generation**:
    *   One SHA-512 hash (64 bytes) is too small for an entire image.
    *   We generate a continuous stream of pseudo-random bytes (Keystream) by hashing a combination of a **Secret Key** and a **Counter**.
    *   `Hash_Block_i = SHA512(Key + i)`s
    *   These blocks are concatenated to match the exact size of the image.

2.  **Encryption (XOR Operation)**:
    *   The original image pixels ($P$) are combined with the keystream bytes ($K$) using the bitwise XOR operation ($\oplus$).
    *   $C = P \oplus K$
    *   This produces the Encrypted Image ($C$).

3.  **Decryption**:
    *   The receiver generates the same keystream using the same Key.
    *   The encrypted image is XORed again with the keystream to recover the original.
    *   $P = C \oplus K$

---

## 2. Security Analysis

### A. Statistical Attack Resistance (The Uniform Histogram)
*   **Observation**: The histogram of the encrypted image is **flat (uniform)**.
*   **Meaning**: Every pixel value from 0 to 255 appears with roughly equal frequency.
*   **Implication**: The encrypted image is statistically indistinguishable from random noise. An attacker analyzing the pixel distribution cannot deduce any information about the original image's shape, edges, or content. This effectively masks the "fingerprint" of the original image.

### B. Confussion and Diffusion
*   **Confusion**: The relationship between the key and the ciphertext is complex due to the non-linear nature of SHA-512.
*   **Diffusion**: The XOR operation with a cryptographically strong keystream ensures that patterns in the plaintext (like large areas of a single color) are completely scattered in the ciphertext.

### C. Correlation Analysis
*   In the original image, adjacent pixels are highly correlated (e.g., a pixel in a blue sky is likely next to another blue pixel).
*   In the SHA-512 encrypted image, the correlation between adjacent pixels drops to near zero. A pixel value gives no prediction about its neighbor.

### D. Key Sensitivity (Avalanche Effect)
*   SHA-512 exhibits a strong Avalanche Effect.
*   Changing a **single bit** of the input Key results in a completely different hash output.
*   Therefore, if an attacker guesses a key that is even slightly wrong, the resulting image will look like random noise, giving no clue that they are "close" to the correct key.

---

## 3. Potential Vulnerabilities

*   **Key Reuse (The "Two-Time Pad" Problem)**: If the same Key is used to encrypt two *different* images, the keystream will be identical. An attacker can XOR the two ciphertexts together to cancel out the keystream:
    *   $C_1 = P_1 \oplus K$
    *   $C_2 = P_2 \oplus K$
    *   $C_1 \oplus C_2 = (P_1 \oplus K) \oplus (P_2 \oplus K) = P_1 \oplus P_2$
    *   The attacker reveals the inputs superimposed on each other. **Solution**: Always use a unique "Nonce" or "IV" combined with the key for every encryption.

*   **Integrity**: This method provides confidentiality (hiding data) but not integrity. An attacker can flip bits in the encrypted image, which will flip the corresponding pixels in the decrypted image, potentially without detection unless a separate MAC (Message Authentication Code) is used.
