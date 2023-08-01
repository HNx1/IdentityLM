# Class that takes in a text string, writes an instruction prompt, then gets logits from a supplied model
# First finds the relevant groups at each token
# Changes the logits to bias one group over another to 
# If we deviate from ideal hash, increase the bias to try and force a close match
# Returns an error if move too far from ideal hash

# Implement a beam search method also to improve the perplexity loss

# Tokenization must be deterministic
import torch
import torch.nn.functional as F
import hashlib
import numpy as np

class IdentitySigner():
    def __init__(self,private_key,public_key,allowed_distance,token_standard,model,tokenizer,device="cuda",maximum_generation=128,digest_size=8):
        self.private_key=private_key # private key exponent
        self.n=public_key[1] # public key modulus
        self.r=public_key[2] # public key 3rd prime
        self.allowed_distance=allowed_distance
        self.token_standard=token_standard
        self.device=device
        self.bias=3.0
        self.bias_add_factor=4.0
        self.maximum_generation=maximum_generation
        self.digest_size=digest_size
        self.model=model
        self.tokenizer=tokenizer
        # Get vocab_size to ensure match between logit shape and vocab_size in decoder - tokenizer vocab_size does not always match the logit shape exactly.
        self.vocab_size=[x for x in model.children()][-1].out_features

        

    def rewrite(self,prompt,temperature=0.5):
        # Tokenize prompt
        token_ids=self.tokenizer.encode(
            prompt, 
            return_tensors="pt")
        # Generate token_standard tokens freely
        token_output = self.model.generate(token_ids, max_length=self.token_standard,return_dict_in_generate=True)
        return_sequence=token_output.sequences[0]
        while return_sequence[-1]!=self.tokenizer.eos_token_id and len(return_sequence)<=self.maximum_generation:

            # Hash tokens generated with public key modulus appended to front
            sequence_to_hash=[bytes(str(self.n),"utf-8")]+[bytes(str(x.item()),"utf-8") for x in return_sequence]
            h=hashlib.shake_256(b"".join(sequence_to_hash))
            # Get a digest of a certain byte size and convert to decimal
            digest=int(h.hexdigest(self.digest_size),16)

            # Get it mod r
            digest=digest % self.r


            # Raise digest to power of private key, mod public key modulus
            digest=pow(digest,self.private_key,self.n)

            digest=digest % self.r

            # Get the binary implementation of this, remove the leading 2 character (which are a binary indicator), get last token_standard character, then left fill with zeroes if needed
            # This is our token group identifier for the next chunk we will generate
            bin_digest=bin(digest)[2:][-self.token_standard:].zfill(self.token_standard)

            # Set initial bias for a chunk
            bias=self.bias
            

            # Set mistakes to zero at start of chunk generation
            mistakes=0

            # For a single chunk, we will generate token_standard tokens
            for char in bin_digest:
            
                # Seed numpy RNG with the current hash, with public key modulus appended to back - we get maximum digest that doesn't overflow numpy manual seeding
                sequence_to_hash=[bytes(str(x.item()),"utf-8") for x in return_sequence]+[bytes(str(self.n),"utf-8")]
                h=hashlib.shake_256(b"".join(sequence_to_hash))
                digest=int(h.hexdigest(4),16)
                np.random.seed(digest)

                # Get logits from single token output from the model
                new_output = self.model.generate(token_ids,decoder_input_ids=return_sequence[None,:], max_length=len(return_sequence)+1,return_dict_in_generate=True,output_attentions=True,output_scores=True)
                logits=new_output.scores[0]/temperature

                # Get token groups from the seeded RNG
                token_group=torch.from_numpy(np.random.choice(self.vocab_size,self.vocab_size//2,replace=False)).to(torch.int64).to(self.device)

                # Build a bias tensor to add to logits, based on the token groups
                bias_locations=torch.zeros(self.vocab_size).scatter_(0, token_group, 1.).to(self.device)

                # Add or subtract bias to logits depending on member of bin_digest we are at
                # This is a little trick for code simplicity, as subtracting logits from one group is equivalent to adding them to the other after the softmax
                logits += bias_locations * bias * (1-2*int(char)) # add if char is 0, subtract if char is 1

                # softmax logits to get probs
                probs=F.softmax(logits,dim=-1)

                # Choose from probs using free RNG (not the seeded one)
                picked_token=torch.multinomial(probs, num_samples=1)[0]

                # Increase bias if we picked from wrong group (cover 2/4 options as we did the add or subtract trick) and add to mistakes
                if picked_token[-1] in token_group:
                    if int(char)==1:
                        # If we were in the first token group, and the char is 1, this is wrong
                        bias+=self.bias_add_factor
                        mistakes+=1
                else:
                    if int(char)==0:
                        # If we were in the second token group, and the char is 0, this is wrong
                        bias+=self.bias_add_factor
                        mistakes+=1

                if mistakes>self.allowed_distance:
                    # If too many mistakes, raise an error - can implement a choice of desired functionality here later
                    print("Too many mistakes in generation")
                    print(mistakes)

                
                # Add picked token to generated ids
                return_sequence=torch.cat((return_sequence,picked_token),dim=-1).to(self.device)

                # Break inner loop if we generate the end sequence token
                if return_sequence[-1]==self.tokenizer.eos_token_id:
                    break
                




        # De-tokenize the generated tokens into text
        return_text=self.tokenizer.decode(return_sequence,skip_special_tokens=True)

        # Return generated text
        return return_text,return_sequence
    
    

