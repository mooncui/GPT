import torch.nn as nn
import math
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import math


class Embedding(nn.Module):
	def __init__(self, voc_size, embedding_dim, seq_length):
		super().__init__()
		self.embedding= nn.Embedding(voc_size,embedding_dim)
		self.pos_embedding= nn.Embedding(seq_length, embedding_dim)
	def forward(self,input_seq):
		pos_ids = torch.arange(len(input_seq),device=input_seq.device)
		word_embeddings = self.embedding(input_seq)
		pos_embeddings = self.pos_embedding(pos_ids)
		return word_embeddings + pos_embeddings

class MultiHead(nn.Module):
	def __init__(self, embedding_dim, q_k_size, v_size, n_head):
		super().__init__()
		self.k = nn.Linear(embedding_dim, n_head*q_k_size)
		self.q = nn.Linear(embedding_dim, n_head*q_k_size)
		self.v = nn.Linear(embedding_dim, n_head*v_size)
		self.n_head = n_head
		self.q_k_size = q_k_size
	def forward(self, inputs): #att_mask):
		q = self.q(inputs) #seq_len, n_head*q_k_size
		#q = q.view(q.size()[0],  self.n_head, self.q_k_size).transpose(0,1) #(head,seq_len,q_k_size)
		k = self.k(inputs) #seq_len, n_head*q_k_size
		#k = k.view(k.size()[0],  self.n_head, self.q_k_size).transpose(0,1).transpose(1,2)
				 
		q_k = torch.matmul(q,k.transpose(0,1))/math.sqrt(self.q_k_size) #seq_len,seq_len
		masked_att = True
		if masked_att:
			mask = torch.tril(torch.ones(inputs.size()[0], inputs.size()[0], device = inputs.device)).bool()
			q_k = q_k.masked_fill_(mask,-1e9)
		#att_mask = att_mask.unsqueeze(1).expand(-1,self.n_head,-1,-1)
		#att = att.masked_fill(att_mask,-1e9)
		q_k = torch.softmax(q_k , dim=-1) # seq_len, seq_len
		v = self.v(inputs) #seq_len, v_size
		q_k_v = torch.matmul(q_k, v) # seq_len, v_size*n_head
		output = q_k_v + inputs 
		#v = v.view(v.size()[0],  self.n_head, self.v_size).transpose(0,1)
		#output = torch.matmul(q_k ,v) #seq_len, v_size
		#return att.reshape(q_k.size()[0], att.size()[1], -1)
		return output

class Attention(nn.Module):
	def __init__(self, embedding_dim, q_k_size, v_size,  n_head):
		super().__init__()
		self.first_norm = nn.LayerNorm(embedding_dim)
		self.att = MultiHead(embedding_dim, q_k_size, v_size, n_head)
		self.proj = nn.Linear(embedding_dim,embedding_dim)

	def forward(self,inputs):
		#att_mask = torch.tril(torch.ones(inputs.size()[1],inputs.size()[1]).unsqueeze(0).unsuqeeze(0)
		#att_mast.expand(-1,self.n_head,-1,-1)

		norm_output = self.first_norm(inputs)
		att_out = self.att(norm_output) #,att_mask)
		#output1 = self.first_linear(output)
		#output1 = self.firt_norm(output + embedding_output)
		#output2 = self.second_linear(output2)
		#output2 = self.second_norm(output2)
		#joined_output = torch.cat((output1,output2),dim=1)
		#joined_output = torch.cat((joined_output,output3),dim=1)
		proj_output = self.proj(att_out)
		output = proj_output + inputs
		return output 

class MLP(nn.Module):
	def __init__(self, embedding_dim, mlp_size):
		super().__init__()
		self.norm = nn.LayerNorm(embedding_dim)
		self.mlp= nn.Sequential(nn.Linear(embedding_dim, mlp_size), nn.ReLU(), nn.Linear(mlp_size, embedding_dim))
	def forward(self, att_output): 
		norm_output = self.norm(att_output)
		mlp_output = self.mlp(norm_output)
		output = mlp_output + att_output 
		return output
	

class Transformer(nn.Module):
	def __init__(self,embedding_dim, q_k_size, v_size,  n_head, mlp_size):
		super().__init__()
		self.att = Attention(embedding_dim, q_k_size, v_size, n_head)
		self.mlp = MLP(embedding_dim, mlp_size)
	def forward(self, inputs):
		att_output = self.att(inputs)
		output = self.mlp(att_output)
		return output 

class NanoGPT(nn.Module):
	def __init__(self,voc_size,embedding_dim, seq_length, q_k_size, v_size,  n_head, mlp_size):
		super().__init__()
		self.emb = Embedding(voc_size,embedding_dim,  seq_length)
		self.tf1 = Transformer(embedding_dim, q_k_size, v_size, n_head, mlp_size)
		self.tf2 = Transformer(embedding_dim, q_k_size, v_size, n_head, mlp_size)
		self.norm = nn.LayerNorm(embedding_dim)
		self.linear_to_logits = nn.Linear(embedding_dim,voc_size)
	def forward(self,token_sequence):
		emb_output = self.emb(token_sequence)
		tf1_output = self.tf1(emb_output)
		tf2_output = self.tf2(tf1_output)
		norm_output = self.norm(tf2_output)
		logits = self.linear_to_logits(norm_output)
		softmax = torch.softmax(logits,dim=-1)
		return softmax

def char_to_int_list(seq: str) -> list:
	mapping = {'a': 0, 'b': 1, 'c': 2}
	return [mapping[char.lower()] for char in seq if char.lower() in mapping]


class SortingDataset(Dataset):
	def __init__(self, num_samples, seq_length=10):
		self.num_samples = num_samples
		self.seq_length = seq_length
		self.chars = ['A', 'B', 'C']
		self.mapping = {'A': 0, 'B': 1, 'C': 2}
		self.index_to_char = {0: 'A', 1: 'B', 2: 'C'}
		
	def __len__(self):
		return self.num_samples
	
	def __getitem__(self, idx):
		input_seq = [random.choice(self.chars) for _ in range(self.seq_length)]
		input_str = ''.join(input_seq)
		sorted_str = ''.join(sorted(input_str))
		input_indices = [self.mapping[c] for c in input_str]
		target_indices = [self.mapping[c] for c in sorted_str]
		return torch.tensor(input_indices, dtype=torch.long), torch.tensor(target_indices, dtype=torch.long)

def test(model):
	num_train_samples = 1000
	dataset = SortingDataset(num_samples=num_train_samples, seq_length=10)
	i = 0
	correct = 0
	while i < 100:
		i = i + 1
		test_input, test_target = dataset[random.randint(0, len(dataset)-1)]
		device = torch.device("cuda")
		test_input = test_input.to(device)  # Use 1D tensor directly
		with torch.no_grad():
			logits = model(test_input)  # (seq_length, voc_size)
			predicted_indices = torch.argmax(logits, dim=-1).cpu().numpy()

		mapping = {0: 'A', 1: 'B', 2: 'C'}
		input_str = ''.join([mapping[i] for i in test_input.cpu().numpy()])
		target_str = ''.join([mapping[i] for i in test_target.cpu().numpy()])
		pred_str = ''.join([mapping[i] for i in predicted_indices])
		if pred_str == target_str:
			correct = correct + 1
			print(f"Correct {correct}/100")
		else:
			print("\nTest Sample:")
			print("Input String: ", input_str)
			print("Sorted Target: ", target_str)
			print("Model Prediction: ", pred_str)

def train():
	# Hyperparameters
	num_epochs = 20 
	batch_size = 1  # Fixed batch size of 1
	learning_rate = 1e-4
	num_train_samples = 1000
	seq_length = 10
	voc_size = 3
	#seq_length = 2048   # Must be greater than the sequence length
	n_head = 3 
	v_size = 16 
	q_k_size = 16 
	embedding_dim = v_size * n_head   # e.g. 16
	mlp_size = embedding_dim * 4 

	# Create dataset and DataLoader (with batch_size=1)
	dataset = SortingDataset(num_samples=num_train_samples, seq_length=seq_length)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


	device = torch.device("cuda")
	model = NanoGPT(voc_size, embedding_dim, seq_length, q_k_size, v_size, n_head, mlp_size).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	#optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)


	model.train()
	for epoch in range(num_epochs):
		total_loss = 0.0
		for batch_inputs, batch_targets in dataloader:
			# Squeeze the batch dimension since batch_size is 1
			inputs = batch_inputs.squeeze(0).to(device)   # (seq_length,)
			targets = batch_targets.squeeze(0).to(device)   # (seq_length,)
			optimizer.zero_grad()
			logits = model(inputs)  # (seq_length, voc_size)
			loss = criterion(logits, targets)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
		avg_loss = total_loss / len(dataloader)
		print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
#	train_data = []
#	for batch_inputs, batch_targets in dataloader:
#		# Squeeze the batch dimension since batch_size is 1
#		inputs = batch_inputs.squeeze(0).to(device)   # (seq_length,)
#		targets = batch_targets.squeeze(0).to(device)   # (seq_length,)
#		train_data.append((inputs,targets))
#	for epoch in range(num_epochs):
#		total_loss = 0.0
#		for (batch_inputs, batch_targets) in  train_data:
#			# Squeeze the batch dimension since batch_size is 1
#			inputs = batch_inputs.squeeze(0).to(device)   # (seq_length,)
#			targets = batch_targets.squeeze(0).to(device)   # (seq_length,)
#			optimizer.zero_grad()
#			logits = model(inputs)  # (seq_length, voc_size)
#			loss = criterion(logits, targets)
#			loss.backward()
#			optimizer.step()
#			total_loss += loss.item()
#		avg_loss = total_loss / len(dataloader)
#		print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
	# After training, randomly pick a sample for testing
	model.eval()
	return model
	#i = 0
	#while i < 100:
	#	i = i + 1
	#	test_input, test_target = dataset[random.randint(0, len(dataset)-1)]
	#	test_input = test_input.to(device)  # Use 1D tensor directly
	#	with torch.no_grad():
	#		logits = model(test_input)  # (seq_length, voc_size)
	#		predicted_indices = torch.argmax(logits, dim=-1).cpu().numpy()

	#	mapping = {0: 'A', 1: 'B', 2: 'C'}
	#	input_str = ''.join([mapping[i] for i in test_input.cpu().numpy()])
	#	target_str = ''.join([mapping[i] for i in test_target.cpu().numpy()])
	#	pred_str = ''.join([mapping[i] for i in predicted_indices])

	#	print("\nTest Sample:")
	#	print("Input String: ", input_str)
	#	print("Sorted Target: ", target_str)
	#	print("Model Prediction: ", pred_str)
	#return model


if __name__ == "__main__":
	#num_epochs = 10 
	#batch_size = 1  # Fixed batch size of 1
	#learning_rate = 1e-3
	#num_train_samples = 1000
	#seq_length = 10
	#voc_size = 3
	##seq_length = 2048   # Must be greater than the sequence length
	#n_head = 3 
	#v_size = 16
	#q_k_size = 16
	#embedding_dim = v_size * n_head   # e.g. 16
	#mlp_size = embedding_dim * 4
	#model = NanoGPT(voc_size, embedding_dim, seq_length, q_k_size, v_size, n_head, mlp_size).to("cpu")
	#test(model)
	model = train()
	test(model)
	#VOC_SIZE = 3 
	#V_SIZE = 16
	#Q_K_SIZE = 16
	#N_HEAD = 3
	#EMBEDDING_DIM = V_SIZE * N_HEAD
	#CONTEXT_LENTH = 10
	#MPL_SIZE = EMBEDDING_DIM* 4
	


	#nano_gpt = NanoGPT(VOC_SIZE, EMBEDDING_DIM, CONTEXT_LENGTH, Q_K_SIZE, V_SIZE, N_HEAD, MPL_SIZE)
	#test_str = "BCAAABCABC"
	#test_tensor = torch.tensor(char_to_int_list(test_str))
	#output = nano_gpt(test_tensor)
	#print(output)
	#indices = torch.argmax(output, dim=1)
	#index_to_char = {0: 'A', 1: 'B', 2: 'C'}
	#decoded_string = ''.join(index_to_char[idx.item()] for idx in indices)
	#print(decoded_string)
	
	
	#embedding_layer = Embedding(VOC_SIZE,EMBEDDING_DIM, CONTEXT_LENGTH)
	#	input_seq = torch.randint(0,VOC_SIZE, (2,10))
	#pos_ids = torch.arange(10).unsqueeze(0).repeat(2,1)
	#embedding_output = embedding_dim_layer(input_seq,pos_ids)	
