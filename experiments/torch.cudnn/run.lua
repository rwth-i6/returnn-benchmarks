require 'hdf5'
require 'torch'
require 'paths'
require 'cudnn'
require 'cutorch'
require 'cunn'
require 'nn'
require 'optim'
require 'LSTM'

function load_chime(train_file_name, options)
	--given dataset is already preprocessed / sliced into 250 time step sequences;
	local train_data = hdf5.open(train_file_name, 'r'):read('/inputs'):all():float()
	local train_classes = hdf5.open(train_file_name, 'r'):read('/targets/data'):all()['classes']
	
	local train_data_batched = torch.reshape(train_data, options.num_batches, options.chunk_size, options.batch_size, options.feat_dim):cuda()
	local train_classes_batched = torch.reshape(train_classes, options.num_batches, options.chunk_size, options.batch_size):cuda()
	
	return train_data_batched, train_classes_batched, options.num_batches
end

function create_model(options)
	cudnn.benchmark = true
	cudnn.fastest = true
	cudnn.verbose = true
       	model = nn.Sequential()
	model:add(cudnn.BLSTM(options.feat_dim, options.hidden_size, options.num_layers))
	model:add(nn.View(options.batch_size * options.chunk_size, options.hidden_size*2))
	model:add(nn.Linear(options.hidden_size*2, options.class_dim))
	model:add(nn.LogSoftMax())
	model:cuda()

	local criterion = nn.ClassNLLCriterion():cuda()

	local params, grads = model:getParameters()

	return {model = model, criterion = criterion, 
		params = params, grads = grads}
end

function main()
	--'Chime3 dataset loading...'
	--chime_train, chime_valid = load_chime(arg[1], arg[2])

	local options = {
				feat_dim = 45,
				hidden_size = 512,
				chunk_size = 250,
				num_layers = 3,
				class_dim = 1501,
				learningRate = 0.001,
				batch_size = 27,
				num_batches = 1011,
				cudnn = 1
			}
	config={
   		learningRate = 0.001
   		--momentum = 0.9
		}		

	
	local xs, ys, size = load_chime(arg[1], options)
	local model = create_model(options)

	timer = torch.Timer()
	file = torch.DiskFile('times.txt', 'w')
	one_timer = torch.Timer()
	for epoch = 1, 10 do
		for i = 1, size do
			one_timer:reset()
			local batch, b_target = xs[i], ys[i]

			--starting closure here:
			local function feval(params_)
				if model.params ~= params_ then
					params:copy(params_)
				end
				model.grads:zero()

				-- forward
				local y_pred = model.model:forward(batch)

				local loss = model.criterion:forward(y_pred, b_target:view(-1))

				--backprop
				local error_signal = model.criterion:backward(y_pred, b_target:view(-1))
				model.model:backward(batch, error_signal)

				print(string.format('epoch = %d, batch = %d, loss = %f, fw/bw time = %f', epoch, i, loss, one_timer:time().real))
				file.writeString(file, string.format('epoch = %d, batch = %d, loss = %f, fw/bw time = %f\n', epoch, i, loss, one_timer:time().real))
				
				return loss, model.grads
			end

			local _, loss_ = optim.sgd(feval, model.params, config)
		end
		print(string.format('ep%d.time=%f\n',epoch, timer:time().real))
		file.writeString(file, string.format('ep%d.time=%f\n',epoch, timer:time().real))
		timer:reset()
	end
end

main()
