include("./exp.jl")

using Unzip

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

            exp_trained_count = 0
            exp_trained_step = 0

            while true

                t._experiences.playGame()

                if exp_trained_count + BATCH_SIZE * BATCH_NUM * TRAIN_EPOCHS < t._experiences.totalCount() * TRAIN_EPOCHS

                    (states, pis, vs) = t._experiences.sampleExperience(BATCH_SIZE * BATCH_NUM)

                    for epoch in 1:TRAIN_EPOCHS

                        @info "Training epoch $(epoch)] started [$(exp_trained_count) / $(t._experiences.totalCount())] ..."

                        for batch in 1:BATCH_NUM

                            state = states[:, :, :, (batch-1)*BATCH_SIZE+1:batch*BATCH_SIZE]
                            pi = pis[:, (batch-1)*BATCH_SIZE+1:batch*BATCH_SIZE]
                            v = vs[:, (batch-1)*BATCH_SIZE+1:batch*BATCH_SIZE]

                            lossF = () -> t._model.loss(state, pi, v)
                            params = t._model.params()
                            grads = gradient(lossF, params)

                            Flux.Optimise.update!(t._opt, params, grads)

                            exp_trained_count += BATCH_SIZE

                        end

                        @info "Training epoch $(epoch)] updated [$(exp_trained_count) / $(t._experiences.totalCount())] ."

                    end

                    exp_trained_step += 1

                    # save model
                    t._model.save("model_$(model.size())_$(exp_trained_step).model")

                end

            end
        end

        return t

    end

end

if abspath(PROGRAM_FILE) == @__FILE__
    model = Model(args["game_size"], channels=args["model_channels"])
    opt = AdamW(args["learning_rate"], (args["adamw_beta1"], args["adamw_beta2"]), args["adamw_weight_decay"])
    train = Train(model, opt)
    train.train()
end
