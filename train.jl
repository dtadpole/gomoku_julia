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

    # attributes
    _model::Model
    _opt::Flux.Optimise.Optimiser

    _experiences::Experiences

    # functions
    size::Function
    train::Function

    # constructor
    function Train(model::Model, opt::Flux.Optimise.Optimiser)

        t = new(model, opt, Experiences(model))

        # load model if exists
        model_filename = "./model/trained_$(model.size())x$(model.size()).curr"
        if isfile(model_filename)
            @info repeat("-", 50)
            @info "Loading Model from [$(model_filename)] ..."
            model.load(model_filename)
            @info "Loaded Model [$(model.size())x$(model.size())] [$(model.channels()) channels] [$(params_size) params]"
        end

        # load optimizer if exists
        opt_filename = "./model/opt_$(model.size())x$(model.size()).opt"
        if isfile(opt_filename)
            @info repeat("-", 50)
            @info "Loading Optimizer from [$(opt_filename)] ..."
            t._opt = load_optimizer(opt_filename)
            @info "Loaded Optimizer [η=$(t._opt[1].eta), β=$(t._opt[1].beta), decay=$(t._opt[2].wd)]"
        end

        # load experiences if exists
        exp_filename = "./model/exp_$(model.size())x$(model.size()).curr"
        if isfile(exp_filename)
            @info repeat("-", 50)
            @info "Loading Experiences from [$(exp_filename)] ..."
            t._experiences.load(exp_filename)
            @info "Loaded Experiences [$(t._experiences.length()), total=$(t._experiences.totalCount()), trained=$(t._experiences.trainedCount())]"
        end

        @info repeat("=", 50)
        t._experiences.startServer()

        if args["model_cuda"] >= 0
            t._opt = t._opt |> gpu
        end

        """Size"""
        t.size = () -> t._model.size()

        """Train"""
        t.train = () -> begin

            BATCH_SIZE = args["train_batch_size"]
            BATCH_NUM = args["train_batch_num"]
            TRAIN_EPOCHS = args["train_epochs"]

            lr_multiplier = 1.0

            while true

                try

                    # t._experiences.playGame() # do not play game in train server.  let the inference servers play games

                    if t._experiences.trainedCount() + BATCH_SIZE * BATCH_NUM * TRAIN_EPOCHS < t._experiences.totalCount() * TRAIN_EPOCHS

                        for epoch in 1:TRAIN_EPOCHS

                            # re-sample each epoch
                            (states, pis, vs) = t._experiences.sampleExperience(BATCH_SIZE * BATCH_NUM)

                            # keeps previous policy
                            prev_policy, _ = t._model.forward(states)

                            # keeps a list of kl divergence
                            kl_list = Vector{Float32}()

                            # keeps a list of loss
                            loss_list = Vector{Float32}()
                            pi_list = Vector{Float32}()
                            v_list = Vector{Float32}()
                            panelty_list = Vector{Float32}()

                            @info "Training epoch [$(epoch)] started [$(t._experiences.length()), total=$(t._experiences.totalCount()), trained=$(t._experiences.trainedCount())] ..."

                            for batch in 1:BATCH_NUM

                                state = states[:, :, :, (batch-1)*BATCH_SIZE+1:batch*BATCH_SIZE]
                                pi = pis[:, (batch-1)*BATCH_SIZE+1:batch*BATCH_SIZE]
                                v = vs[:, (batch-1)*BATCH_SIZE+1:batch*BATCH_SIZE]

                                # convert from visit count to distribution
                                pa = softmax(log.(pi .+ 1e-8), dims=1)

                                lossF = () -> t._model.loss(state, pa, v)[1]  # [IMPORTANT] use normalized distribution pa !!
                                params = t._model.params()
                                grads = gradient(lossF, params)

                                Flux.Optimise.update!(t._opt, params, grads)

                                t._experiences.addTrainedCount(BATCH_SIZE)

                                # calculate new policy
                                new_loss, loss_pi, loss_v, loss_panelty, new_pi, new_v = t._model.loss(state, pa, v)

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
                                    @info "KL divergence [$(round(kl_mean, digits=4)) > 4 * $(args["train_kl_target"])] ... stopping early ."
                                    break
                                end

                            end

                            loss_epoch = round(mean(loss_list), digits=4)
                            pi_epoch = round(mean(pi_list), digits=4)
                            v_epoch = round(mean(v_list), digits=4)
                            panelty_epoch = round(mean(panelty_list), digits=4)

                            @info "Training epoch [$(epoch)] updated [$(t._experiences.length()), total=$(t._experiences.totalCount()), trained=$(t._experiences.trainedCount())] [loss : $(loss_epoch) = ($(pi_epoch),pi) + ($(v_epoch),v) + ($(panelty_epoch),panelty)]"
                            # @info loss_list pi_list v_list panelty_list

                            # check for kl divergence
                            kl_epoch_mean = mean(kl_list)

                            if kl_epoch_mean > args["train_kl_target"] * 2.0 && lr_multiplier > 1.0 / args["learning_rate_range"]
                                lr_multiplier /= 1.5
                                new_learning_rate = args["learning_rate"] * lr_multiplier
                                @info "KL divergence [$(round(kl_epoch_mean, digits=4)) > $(args["train_kl_target"]) * 2.0] ... reducing learning rate to [$(round(new_learning_rate, digits=5))] ."
                                # update learning rate
                                t._opt[1].eta = new_learning_rate
                            elseif kl_epoch_mean < args["train_kl_target"] / 2.0 && lr_multiplier < args["learning_rate_range"]
                                lr_multiplier *= 1.5
                                new_learning_rate = args["learning_rate"] * lr_multiplier
                                @info "KL divergence [$(round(kl_epoch_mean, digits=4)) < $(args["train_kl_target"]) / 2.0] ... increasing learning rate to [$(round(new_learning_rate, digits=5))] ."
                                # update learning rate
                                t._opt[1].eta = new_learning_rate
                            else
                                new_learning_rate = args["learning_rate"] * lr_multiplier
                                @info "KL divergence [$(round(kl_epoch_mean, digits=4)) within range] ... keeping learning rate as [$(round(new_learning_rate, digits=5))] ."
                            end

                        end

                        # print separator
                        @info repeat("-", 50)

                        # save model
                        @info "Saving model to [$(model_filename)] ..."
                        backup_file(model_filename)
                        t._model.save(model_filename)

                        # save optimizer
                        @info "Saving optimizer to [$(opt_filename)] ..."
                        backup_file(opt_filename)
                        save_optimizer(opt_filename, t._opt)

                        # save experiences
                        @info "Saving experiences to [$(exp_filename)] ..."
                        backup_file(exp_filename)
                        t._experiences.save(exp_filename)

                        # print separator
                        @info repeat("-", 50)

                    else

                        @info "Training is waiting for more experiences [$(t._experiences.length()), total=$(t._experiences.totalCount()), trained=$(t._experiences.trainedCount())] ..."

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
    # model
    model = Model(args["game_size"], channels=args["model_channels"])
    params_size = sum([length(l) for l in Flux.params(model._model)])
    @info "[Model] [$(model.size())x$(model.size())] [$(model.channels()) channels] [$(params_size) params]"
    # optimizer
    opt = AdamW(args["learning_rate"], (args["adamw_beta1"], args["adamw_beta2"]), args["adamw_weight_decay"])
    @info "[Optimizer] [η=$(opt[1].eta), β=$(opt[1].beta), decay=$(opt[2].wd)]"
    # start training
    train = Train(model, opt)
    train.train()
end
