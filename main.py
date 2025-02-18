import pytorch as nn



class Embedding(nn.Module):
	def __init__(self, voc_size, embeding_dim, context_length):
		self.embeding = nn.Embedding(voc_size,embeding_dim)
		self.pos_embeding = nn.Embedding(context_length, embeding_dim)
	def forward(self,input_seq, pos_ids):
		word_embeddings = self.embeding(input_seq)
		pos_embeddings = self.pos_embeding(pos_ids)
		return word_embeddings + pos_embeddings

class Head(nn.Module):
	def __init__(self, embedding_dim, q_k_size, v_size, n_head):
		self.k = nn.Linear(embedding_dim, n_head*q_k_size)
		self.q = nn.Linear(embedding_dim, n_head*q_k_size)
		self.v = nn.Linear(embeding_dim, n_head*v_size)
		self.n_head = n_head
	def forward(self, embeding_output, att_mask):
		q = self.q(embeding_output)
		q = q.view(q.size()[0], q.size()[1], self.n_head, self.q_k_size).transpose(1,2) #(batch_size,head,seq_len,q_k_size)
		k = self.k(embeding_output)
		k = k.view(k.size()[0], q.size()[1], self.n_head, self.q_k_size).transpose(1,2).transpose(2,3)
		att = torch.matmul(q,k)/math.sqrt(self.q_k_size)
		att_mask = att_mask.unsqueeze(1).expand(-1,self.n_head,-1,-1)
		att = att.masked_fill(att_mask,-1e9)
		att = torch.softmax(att,dim=-1)
		v = self.v(embeding_output)
		v = v.view(v.size()[0], v.size()[1], self.n_head, self.v_size).transpose(1,2)
		att = torch.matmul(att,v) #batch_size, seq_len, v_size
		return att.reshape(att.size()[0], att.size()[1], -1)

class Attention(nn.Module):
	def __init__(self, embedding_dim, q_k_size, v_size, full_con_nn_size, n_head):
		self.first_norm = nn.LayerNorm(embeding_dim)
		self.first_att = Head(embedding_dim, q_k_size, v_size, n_head)
		self.second_att = Head(embedding_dim, q_k_size, v_size, n_head)
		self.third_att = Head(embedding_dim, q_k_size, v_size, n_head)
		self.second_norm = nn.LayerNorm(embeding_dim)

		self.full_con = nn.Sequencial(nn.Linear(embedding_size, full_con_nn_size), nn.ReLU(), nn.Linear(f_size, embedding_size))
		self.third_norm = nn.LayerNorm(embedding_dim)

	def forward(self,embeding_output):
		att_mask = torch.tril(torch.ones(embeding_output.size()[1],embeding_output.size()[1]).unsqueeze(0).unsuqeeze(0)
		att_mast.expand(-1,self.n_head,-1,-1)

		embedding_output = self.first_norm(embedding_output)
		output1 = self.first_att(embedding_output,att_mask)
		#output1 = self.first_linear(output)
		#output1 = self.firt_norm(output + embedding_output)
		output2 = self.second_att(embedding_output,att_mask)
		output3 = self.third_att(embedding_output,att_mask)
		#output2 = self.second_linear(output2)
		#output2 = self.second_norm(output2)
		#joined_output = torch.cat((output1,output2),dim=1)
		joined_output = torch.cat((joined_output,output3),dim=1)
		joined_output = self.second_norm(joined_output)
		output = self.third_norm(self.full_con(joined_output))

		
		
		
		
	

		
		


def softmax(decoding_output_seq):
	return nn.softmax(decoding_output_seq):

#def softmax(logits: List[float]): List[float]
#    exps = [math.exp(x) for x in logits]
#    summ = sum(exps)
#    return [exp_val/summ for exp in exps]

if __name__ == __Main__:
	VOC_SIZE = 20000
        EMBEDDING_DIM = 128
        CONTEXT_LENGTH = 2048
	embedding_layer = Embedding(VOC_SIZE,EMBEDDING_DIM, CONTEXT_LENGTH)
        input_seq = torch.randint(0,VOC_SIZE, (2,10))
	pos_ids = torch.arange(10).unsqueeze(0).repeat(2,1)
	embedding_output = embeding_layer(input_seq,pos_ids)	
