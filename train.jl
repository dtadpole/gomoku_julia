include("./exp.jl")

using Unzip

function save_optimizer(path::String, opt::Flux.Optimise.Optimiser)
    open(path, "w") do io
        serialize(io, opt)
    end
end

function load_optimizer(path::String)
    open(path, "r") do io
        return deserialize(io)
    end
end

mutable struct Train

    _experiences::Experiences

    # functions
    size::Function
    modelPath::Function
    expPath::Function
    optPath::Function
    train::Function

    # constructor
    function Train(id_list::Vector{Int})

        t = new()

        """Size"""
        t.size = () -> args["game_size"]

        """Model path"""
        t.modelPath = (id::Int; tag = "curr") -> begin
            model_path = "./trained/$(id)/model_$(t.size())x$(t.size()).$(tag)"
            if !isdir(dirname(model_path))
                mkpath(dirname(model_path))
            end
            return model_path
        end

        """Opt path"""
        t.optPath = (id::Int) -> begin
            opt_path = "./trained/$(id)/opt_$(t.size())x$(t.size()).curr"
            if !isdir(dirname(opt_path))
                mkpath(dirname(opt_path))
            end
            return opt_path
        end

        """Exp path"""
        t.expPath = (id::Int) -> begin
            exp_path = "./trained/$(id)/exp_$(t.size())x$(t.size()).curr"
            if !isdir(dirname(exp_path))
                mkpath(dirname(exp_path))
            end
            return exp_path
        end

        # initialize models, optimizers
        model_list = Vector{Model}()
        opt_list = Vector{Flux.Optimise.Optimiser}()

        for id in id_list

            # initialize model
            model = Model(args["game_size"], channels=args["model_channels"])
            params_size = sum([length(l) for l in Flux.params(model._model)])
            @info "[$(id)] Initialize Model [$(model.size())x$(model.size())] [channels=$(model.channels())] [params=$(params_size)]"
            # load model if exists
            model_filename = t.modelPath(id)
            if isfile(model_filename)
                @info repeat("-", 50)
                @info "[$(id)] Loading Model from [$(model_filename)] ..."
                model.load(model_filename)
                @info "[$(id)] Loaded Model [$(model.size())x$(model.size())] [channels=$(model.channels())] [params=$(params_size)]"
            end
            push!(model_list, model)

            # initialize optimizer
            opt = AdamW(args["learning_rate"], (args["adamw_beta1"], args["adamw_beta2"]), args["adamw_weight_decay"])
            @info "[$(id)] Initialize Optimizer [η=$(round(opt[1].eta, digits=5)), β=$(round.(opt[1].beta, digits=4)), decay=$(opt[2].wd)]"
            # load optimizer if exists
            opt_filename = t.optPath(id)
            if isfile(opt_filename)
                @info repeat("-", 50)
                @info "[$(id)] Loading Optimizer from [$(opt_filename)] ..."
                opt = load_optimizer(opt_filename)
                @info "[$(id)] Loaded Optimizer [η=$(round(opt[1].eta, digits=5)), β=$(round.(opt[1].beta, digits=4)), decay=$(opt[2].wd)]"
            end
            # move to gpu
            if args["model_cuda"] >= 0
                opt = opt |> gpu
            end
            push!(opt_list, opt)

        end

        # initialize experiences
        t._experiences = Experiences(model_list, opt_list)

        for id in id_list
            # load experiences if exists
            exp_filename = t.expPath(id)
            if isfile(exp_filename)
                @info repeat("-", 50)
                @info "[$(id)] Loading Experiences from [$(exp_filename)] ..."
                t._experiences.load(id, exp_filename)
                @info "[$(id)] Loaded Experiences [len=$(t._experiences.length()), total=$(t._experiences.totalCount()), trained=$(t._experiences.trainedCount())]"
            end
        end

        # start experiences server
        @info repeat("=", 50)
        t._experiences.startServer()

        """Train"""
        t.train = () -> begin

            BATCH_SIZE = args["train_batch_size"]
            BATCH_NUM = args["train_batch_num"]
            TRAIN_EPOCHS = args["train_epochs"]

            while true

                try

                    for id in id_list

                        # t._experiences.playGame() # do not play game in train server.  let the inference servers play games

                        if t._experiences.trainedCount(id) + BATCH_SIZE * BATCH_NUM * TRAIN_EPOCHS < t._experiences.totalCount(id) * TRAIN_EPOCHS

                            for epoch in 1:TRAIN_EPOCHS

                                # re-sample each epoch
                                (states, pis, vs) = t._experiences.sampleExperience(id, BATCH_SIZE * BATCH_NUM)

                                # keeps previous policy
                                prev_policy, _ = t._experiences.model(id).forward(states)

                                # keeps a list of kl divergence
                                kl_list = Vector{Float32}()

                                # keeps a list of loss
                                loss_list = Vector{Float32}()
                                pi_list = Vector{Float32}()
                                v_list = Vector{Float32}()
                                panelty_list = Vector{Float32}()

                                @info "[$(id)] Training epoch [$(epoch)] started [len=$(t._experiences.length()), total=$(t._experiences.totalCount()), trained=$(t._experiences.trainedCount())] ..."

                                for batch in 1:BATCH_NUM

                                    state = states[:, :, :, (batch-1)*BATCH_SIZE+1:batch*BATCH_SIZE]
                                    pi = pis[:, (batch-1)*BATCH_SIZE+1:batch*BATCH_SIZE]
                                    v = vs[:, (batch-1)*BATCH_SIZE+1:batch*BATCH_SIZE]

                                    # convert from visit count to distribution
                                    pa = softmax(log.(pi .+ 1e-8), dims=1)

                                    lossF = () -> t._experiences.model(id).loss(state, pa, v)[1]  # [IMPORTANT] use normalized distribution pa !!
                                    params = t._experiences.model(id).params()
                                    grads = gradient(lossF, params)

                                    Flux.Optimise.update!(t._experiences.opt(id), params, grads)

                                    t._experiences.addTrainedCount(id, BATCH_SIZE)

                                    # calculate new policy
                                    new_loss, loss_pi, loss_v, loss_panelty, new_pi, new_v = t._experiences.model(id).loss(state, pa, v)

                                    # keep track of loss
                                    push!(loss_list, new_loss |> cpu)
                                    push!(pi_list, loss_pi |> cpu)
                                    push!(v_list, loss_v |> cpu)
                                    push!(panelty_list, loss_panelty |> cpu)

                                    # extract prev policy
                                    prev_pi = prev_policy[:, (batch-1)*BATCH_SIZE+1:batch*BATCH_SIZE]

                                    # calculate KL divergence
                                    kl = reshape(sum(prev_pi .* (log.(prev_pi) - log.(new_pi)), dims=1), BATCH_SIZE)
                                    kl_mean = mean(kl)

                                    # update global list
                                    kl_list = vcat(kl_list, kl |> cpu)

                                    # check if kl divergence is too high, break early if so
                                    if kl_mean > 4 * args["train_kl_target"]
                                        @info "[$(id)] KL divergence [$(round(kl_mean, digits=4)) > 4 * $(args["train_kl_target"])] ... stopping early ."
                                        break
                                    end

                                end

                                loss_epoch = round(mean(loss_list), digits=4)
                                pi_epoch = round(mean(pi_list), digits=4)
                                v_epoch = round(mean(v_list), digits=4)
                                panelty_epoch = round(mean(panelty_list), digits=4)

                                @info "[$(id)] Training epoch [$(epoch)] updated [loss : $(loss_epoch) = ($(pi_epoch),pi) + ($(v_epoch),v) + ($(panelty_epoch),panelty)]"
                                # @info loss_list pi_list v_list panelty_list

                                # check for kl divergence
                                kl_epoch_mean = mean(kl_list)

                                if kl_epoch_mean > args["train_kl_target"] * 2.0
                                    new_learning_rate = max(t._experiences.opt(id)[1].eta / 1.5, args["learning_rate"] / args["learning_rate_range"])
                                    @info "[$(id)] KL divergence [$(round(kl_epoch_mean, digits=4)) > $(args["train_kl_target"]) * 2.0] ... reducing learning rate to [$(round(new_learning_rate, digits=5))] ."
                                    # update learning rate
                                    t._experiences.opt(id)[1].eta = new_learning_rate
                                elseif kl_epoch_mean < args["train_kl_target"] / 2.0
                                    new_learning_rate = min(t._experiences.opt(id)[1].eta * 1.5, learning_rate, args["learning_rate"] * args["learning_rate_range"])
                                    @info "[$(id)] KL divergence [$(round(kl_epoch_mean, digits=4)) < $(args["train_kl_target"]) / 2.0] ... increasing learning rate to [$(round(new_learning_rate, digits=5))] ."
                                    # update learning rate
                                    t._experiences.opt(id)[1].eta = new_learning_rate
                                else
                                    learning_rate = t._experiences.opt(id)[1].eta
                                    @info "[$(id)] KL divergence [$(round(kl_epoch_mean, digits=4)) within range] ... keeping learning rate as [$(round(learning_rate, digits=5))] ."
                                end

                            end

                            # print separator
                            @info repeat("-", 50)

                            # save model
                            model_ = t._experiences.model(id)
                            param_size = sum(length.(Flux.params(model_)))
                            model_filename = t.modelPath(id)
                            @info "[$(id)] Saving model to [$(model_filename)] : [$(model_.size())x$(model_.size())] [channels=$(model_.channels())] [params=$(param_size)] ..."
                            backup_file(model_filename)
                            model_.save(model_filename)

                            # save optimizer
                            opt_ = t._experiences.opt(id)
                            opt_filename = t.optPath(id)
                            @info "[$(id)] Saving optimizer to [$(opt_filename)] : [η=$(round(opt_[1].eta, digits=5)), β=$(round.(opt_[1].beta, digits=4)), decay=$(opt_[2].wd)] ..."
                            backup_file(opt_filename)
                            save_optimizer(opt_filename, opt_)

                            # save experiences
                            exp_filename = t.expPath(id)
                            @info "[$(id)] Saving experiences to [$(exp_filename)] : [len=$(t._experiences.length(id)), total=$(t._experiences.totalCount(id)), trained=$(t._experiences.trainedCount(id))] ..."
                            backup_file(exp_filename)
                            t._experiences.save(id, exp_filename)

                            # print separator
                            @info repeat("-", 50)

                        else

                            @info "[$(id)] Training is waiting for more experiences [len=$(t._experiences.length(id)), total=$(t._experiences.totalCount(id)), trained=$(t._experiences.trainedCount(id))] ..."

                        end

                    end

                catch e

                    @error "Error" exception = (e, catch_backtrace())

                finally

                    sleep(15)

                end

            end

        end

        return t

    end

end

if abspath(PROGRAM_FILE) == @__FILE__
    @info repeat("=", 50)
    # train model[1] and model[2]
    train = Train([1, 2])
    train.train()
end
