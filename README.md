# IdentityLM - cryptographic identity proof via language model output

- IdentityLM is a method of cryptographic proof of identity that is encoded
  directly into language model output (this has nothing to do with
  cryptocurrency just be clear). It essentially implements public key
  cryptography directly into the token generation of a model - any language
  model can be used, but for practical reasons discussed later it is likely best
  to separate generation and encryption models and apply them sequentially.
- This method (as implemented as a starting point in this repo) allows you to
  produce natural language output, that is cryptographically signed to be shown
  to come from you. The text itself carries the encryption - no added hashes,
  keywords or metadata (except the generating model tokenizer list, which I hope
  to standardise as part of this project) are needed.
- I'm looking for help to develop this from the starting point here, and calling
  for opinions or contributions from the wider community. We also need more
  expert scrutiny on the security of this method (which really just slightly
  varies existing widely used standards) to ensure there are no holes and to
  have a very strong understanding of the encryption strength (particularly as
  the encryption strength varies significantly with input text length and
  certain parameter choices).
- I'm committed to building this as an open standard where the encryption and
  detection methods are exposed for everyone to see and verify. The strength
  comes from the hidden private key, the difficult of factorising arbitrarily
  large primes, and the security of hash functions. My methods just embed this
  security into the distributions of language models.

### Important Disclaimers

- Please note: this repo as it stands today SHOULD NOT be used for cryptographic
  signing of text with any expectation of security. The current implementation
  should be seen as a PROOF OF CONCEPT.

- Also note this repo has been implemented in a way that prioritizes simplicity
  to maximise ease-of-understanding. This means relatively inefficient for-loops
  where vectorisation could be used, inefficient model inference (no KV caching
  even though we generate one token at a time), single sequence generation (no
  batching) and key gen directly in the repo, rather than a secure third party
  cryptographic standard. If there is sufficient interest in this idea from the
  community, I will rewrite all of these things to be production ready - and of
  course I would be super grateful for any contributors who want to build
  production code for this.
- For the above reason, the repo is also not currently suitable for attempted
  detection at large token standards.
- The model being used in the demo is small (so that people can try it out
  easily) and is instruction trained for paraphrasing which is not exactly the
  task at hand and therefore will produce some pretty weird generations. A
  bigger model, trained specifically for this task, with extensive beam search,
  will produce far better generations. Right now there is a lot of work to do on
  the generation side.
- The goal of this is not to replace current encryption methods. Rather it
  should work alongside them, giving further security, as well as the direct
  applications listed below.

### How strong is the encryption potential

- In this repo we are using both n-bit public key encryption, where breaking
  involves factorizing an n-bit product of two primes, and m-bit "hash"
  encryption, where breaking involves finding a pseudorandom m-bit number
  (clearly much harder for m=n). To illustrate the practical difference, it's
  computationally feasible today to break 512-bit (but not 1024-bit) public key
  encryption, and analogously 64-bit (but not 128-bit) for "hash" encryption.
- Under a 2 prime (i.e. standard) RSA public key implementation the limiting
  factor here is the public key side. Right now, 256 tokens can be encrypted
  with a 256-bit RSA standard - not very strong. However, there is low hanging
  fruit (by implementing aggressive beam search and using more token groups) to
  push this to 512-bit, and probably even 768 or 1024, with limited model
  performance loss. So basically in the next couple of months I expect to get
  this to where 256 tokens can be encrypted in a way that is computationally
  infeasible to break, under the existing well supported RSA 2-prime standard.
- However, to solve this RSA limiting issue, I developed an adaptation of
  3-prime RSA that I believe allows arbitrary RSA encryption for any token
  sequence. In this case, the limiting factor becomes the actual "hash"
  encryption of the token sequence. For 32 tokens, this would be 32-bit hash
  encryption, which similarly to above could be probably be pushed to 128-bit
  with relatively limited loss in model performance. Therefore, if my slight
  adaptation of RSA is secure, it would allow just 32 tokens to be encrypted at
  a level that is computationally infeasible to break. The repo as it stands
  today uses this 3 prime implementation, which I describe in more detail below.
- To summarise, under regular RSA 256 tokens are needed to be impossible to
  break, under my adaptation 32 tokens are needed - but that adaptation needs to
  be carefully checked for security!

### Applications

- Send an email or text message, having a language model rewrite it , and then
  the recipient being able to say with certainty that it came from you purely
  based on the text
- Create universal internet identities (if the user wants!) and easily move
  identity and social trust from site to site. Are you a big contributor on
  Stack Overflow? Automatically transfer that trust to any other forum purely
  through text generation.
- Any official communication can be cryptographically shown to come from a
  specific company or organisation. Even a phone call, so long as the script is
  run through an LM, can be automatically shown to actually be the company in
  question, not a scam.
- Counter political deepfakes (potentially a huge problem in the future) by
  changing just a few words of your speech with a language model. In general,
  it's a powerful method to disprove deepfakes of public figures/celebrities.
- This method naturally extends to image/video - by slightly altering an image
  in a natural way we can prove it came from a specific source

### Key advantages

- The detection doesn't need the LM itself. It just needs the model tokenizer
  and public key of the model user, and is very computationally quick relative
  to the cost of a brute force attack. To give a complexity analogy, detection
  is like deciphering an encrypted message sent to you using your own private
  key, whereas creating a message that fools the detector is like brute forcing
  a private key.
- A relatively small model can learn to rewrite text effectively, so the
  adaptation can be done at little computational cost. However, this method can
  work on any model with a public tokenizer list. That said, I expect extensive
  beam search (basically looking n tokens ahead in generation) to be used quite
  a lot in encryption, so I think it's better to separate the generation model
  (which can be a very large model like GPT-4) from the encryption model, and
  just use the encryption model to rewrite the generation of the large model. Of
  course you can also write your own messages and pass them into the encryption
  model.
- The method is flexible in encryption strength - which allows even very short,
  relatively unimportant text to be weakly encrypted.
- An attacker changing a single word even close to the end of a text sequence
  will throw off the entire subsequent detection. In order to add text, an
  attacker must append to an existing valid generation, and must solve the
  cryptographic problem to do that addition.

### Background

- Recently a [https://arxiv.org/pdf/2301.10226.pdf](paper) about watermarking LM
  output was awarded outstanding paper at ICML, which brought it to my
  attention. It's a very interesting paper - I highly recommend reading it as it
  basically explains the logic behind the logit biasing and token grouping, and
  also why I'm excited about minimal perplexity loss and the potential of beam
  search.
- The paper focuses on distinguishing model generated vs human content - how can
  OpenAI for example mark GPT-4 output so that they can find it in the future?
  The logit biasing they use really piqued my interest and I started thinking
  about what kinds of statistical patterns could be encoded this way - which led
  me to cryptography.
- This paper helped crystallise some thoughts I was having about model
  distribution biasing and it's really a foundation for IdentityLM.

### Repo info

- What this repo currently does is allow you to rewrite a text sequence, with
  limited changes, to prove cryptographically that it came from you. It uses a
  small paraphrasing encoder-decoder that I just pulled from Hugging Face, so
  generations may look a bit wonky at times.
- demo.ipynb contains a basic implementation showing this functionality. Write
  whatever you want (subject to the context window of the model), and you should
  get back something reasonably similar. However, the detector function of the
  repo can then tell you which one is encrypted. Generate new keys, and it can
  distinguish all 3 generations. Note this demo, to allow for very short inputs,
  uses a weak "8-token" encryption standard - enough that manual generation
  should pretty consistently work, but highly vulnerable to brute force.
- The exceptions are low entropy sequences - short runs of text where each next
  token is highly determined by the previous ones - these are difficult to
  cryptographically sign. However, I believe that a promising method to assist
  many low entropy sequences is by raising the underlying model temperature
  slightly - to be explored. This model temperature idea also helps in
  sequential generation when the model is struggling to produce sequences within
  good bounds.
- For example, if you enter "Why did the chicken cross the road?" It's unlikely
  the model can rewrite that in a cryptographically significant fashion. A
  function that automatically detects the suitability of the input text for the
  supplied encryption parameters is on my to-do list also.

### Adapted 3 prime RSA

- The key idea of this adapted method concerns multi-prime RSA in the setting
  where 1 or more of the primes is known, and indeed the research I reviewed
  suggests it's still secure, particularly if the unknown primes are very large.
- In this method, we pick two arbitrary size primes p and q. We also pick a
  token_standard bit prime r, subject to it being close to 2^k (I use at least
  90% in this repo, but we could go much closer). r is published as part of the
  public key.
- The idea of this is that any encrypted hash can be reduced mod r, while
  preserving the message under decryption. This is due to Fermat's Little
  Theorem. In this repo we have to reduce mod 2^k as we are cutting the hashes
  to k-bit size. However, if we reduce mod r first, then the hash is the same
  mod 2^k, as r is less than 2^k. The weakness of this is that hashes between r
  and 2^k can no longer be produced, and small hashes have double the
  likelihood. For this reason we ensure r is close to 2^k.
- The question being - is there an attack vector I am missing here, either from
  the mod r reduction of the hash (my understanding is a big integer library
  which lets us take a lot of bytes in the digest makes this very secure), or
  through the public nature of r? Please do weigh in!

### The details of the method

- The core challenge is that in order for the watermark paper method to work,
  the detector must have knowledge of which tokens should be biased for the next
  token generation
- Unfortunately this means that generation that simulates a user can easily be
  done using the detector. The watermark paper method is really designed for a
  model creator to track anything generated by that model, because they keep the
  detector RNG hidden so an outsider can't replicate the groups at each state.
- To solve this, we will lever ideas of public key cryptography and hash
  functions
- Core idea: Across the entire sequence, anyone with the public key can still
  figure out the groups at any stage. Our encryption will instead decide which
  group is biased at each stage and closely replicate private key encrypted
  hashes at fixed intervals using this statistical bias. The detector will then
  use the public key to try and verify a correct encryption similar to the
  actual produced sequence.
- Language models learn a conditional next token distribution. This token
  distribution is generated by performing a softmax operation on logits,
  produced at the end of the model. If we add 1 to the logit of a token, it's
  roughly equivalent to multiplying the probability of that token by e (very
  slightly less, as the normalization demoninator in softmax also increases). If
  we subtract 1, it's roughly equivalent to dividing the probability by e.
  Therefore, we can add to logits to change the probability distribution that is
  produced, biasing the tokens whose logits we added to.
- In the watermarking paper above, they describe how this logit addition can be
  used to watermark a particular model - a unique hash can be generated by the
  model creator that changes the token distribution to bias a deterministic
  group of tokens (say 50% or 25% of the tokens). This creates a statistical
  signature of the token group in question, that can be later hypothesis tested
  to see if the watermark is present in text or not.
- In this case, rather than watermarking a model, we want to watermark a user in
  a way that is very hard to replicate, but easy to verify by a different user,
  with as limited information as possible about the generation circumstances of
  the text.
- Let's walk through signing and detection.
- Let K be the token standard, i.e. the length of each hash we will look to
  replicate.
- Freely generate K token, then hash them with the public key modulus appended
  to the front, and raise this to the power of the private key. Reduce this mod
  r (the third prime that we make public). This gives a K-bit binary number, as
  r is less than 2^K.
- Each character in this binary sequence tells us which group to bias in the
  next K tokens we generate. This basically will encode into those tokens a
  K-bit hash that is the resulting of encrypting a deterministic message with
  our private key. This hash then requires either breaking the hash directly, or
  breaking the private key, to replicate. But the actual result we get in the
  tokens actually generated will be similar to this encrypted hash as we are
  directly biasing it.
- For each token we hash the entire token sequence sequence with the private key
  modulus appended to the back. We use this to seed a random number generator to
  pick the token groups at that stage.
- Generate the logits for that token from the model, and add to the relevant
  token group a bias. We apply temperature before the bias. Then pick a token
  from this and add it to our generated sequence. During the generation of a K
  length chunk of tokens, we'll track how far we are from the ideal hash, and
  increase the bias to ensure we stay close.
- Exit when we pick the eos token.
- When it comes to detection, we tokenize our text and divide it into chunks of
  length K - right now we discard the remaining end tokens, but I plan to use
  them in the future to increase security.
- Ignore the first chunk, as this was generated freely.
- For each chunk we can recover the exact token groups used in the signing, as
  all we need is the public key modulus and the previous token sequence.
  Therefore, for each chunk we can write a K-bit binary number indicating which
  group the generation was in. We take that binary number and raise it to the
  power of the public key exponent. This will recover the original chunk hash
  generated. If these match, then we've detected that the chunk was encrypted.
  Now remember we statistically generated the tokens, so we also need to check
  similar (i.e. that agree in all but M places) binary numbers to the one we
  actually had in the generation as the signing process may not have exactly
  matched the ideal binary sequence.
- In signed text, what we see is that every chunk is close enough to the ideal
  hash to quickly find it. In unsigned text, only some chunks will be close. And
  indeed, as we ramp up K it will be statistically improbable than even 1
  unsigned chunk will match. The watermark paper contains some incredible
  z-scores in their hypothesis around how statistically strong this signalling
  can be.
- Now the similarity checking is computationally not that onerous relative to
  breaking the hash by random search. In a 64-bit standard where we allow 10
  deviations from ideal (and I believe we can do many fewer with low perplexity
  loss), there is 10^9 asymmetry in computation time between verification and
  breaking the hash. And with larger K it's even bigger.

### Standards proposal

- Model publishers should publish their tokenizer (only indices of tokens are
  needed, not log frequencies as you see in SentencePiece vocabs for example),
  as this knowledge is needed for encryption and detection
- This suggests a new three tiered standard to classify models in this context -
  "closed" models where everything is hidden, "open" models where everything is
  shown, and "identity-open" models that publish a tokenizer but not weights.
  The latter two types can be used for this cryptographic identity proof.
- A user should be able to generate a user public key, and a user private key,
  from the trusted IdentityLM source. Users should be able to regularly
  regenerate these keys. We will establish a centralized system where public
  keys and real user identity can be strongly linked, if the user wishes. This
  will help organizations, governments, prominent individuals in any community
  etc. to have a simple way to prove that content of any kind comes from them.
- This up-to-date central repository at IdentityLM will be used to build a
  detection API that allow internet providers, phone companies, email services
  and really any company or organization interested in identity proof to
  integrate automatic identity detection into their products so that users can
  easily validate the identity of the creator of any content they see
- Model APIs should allow logit level outputs, so that users can
  cryptographically sign their generation, or should allow the input of a user
  private key into their user interface that then changes the logits under the
  hood (with appropriate security around the private key entry - could be
  provided by IdentityLM).
- Currently, the detector requires knowledge of the model to detect (i.e. it
  must know the entire tokenizer to detect). IdentityLM should publish an open
  set of tokenizers that can be used as standard in encryption models, and then
  embedded in the detection function so no tokenizer loading is needed. Due to
  need for very precise and deterministic tokenization, I believe encryption
  models should only focus on putting the encryption into words and numbers,
  with special characters and punctuation carrying no signal.
- IdentityLM will train a relatively small model, trained to rewrite text with a
  few small changes. This model can then be flexibly used by the community to
  encrypt any text they wish. Currently, good small paraphrasing models are all
  encoder-decoder style, which makes for complex generation, and are trained for
  a paraphrasing, a slightly different task from what we want to accomplish.

### Contact

- Any feedback, analysis, opinions much appreciated!
- If you want to talk about IdentityLM outside of just raising a Github issue
  feel free to email me at aeldarion587@gmail.com

### Citation

- Cite as: HNx1, (Aug 2023), Cryptographic identity proof via language model
  output. https://github.com/HNx1/IdentityLM
