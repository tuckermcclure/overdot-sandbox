# overdot-sandbox
A place to work on ideas for a simulation engine in Julia

Hi. If you're reading this, I've probably talked with you about simulation ideas, and we agreed that we wanted there to be an open-source tool for simulating modular systems of systems with a minimum of fuss. We've probably talked about how to make things easy and fast, how to make things scale nicely, and how to keep boilerplate down. I think we've stirred this pot enough that it's time for a first draft. In this repo, I've sketched out a few ideas building on the conversations we've had and code snippets we've exchanged. What's here is already my favorite simulation engine for its simplicity, and yet I think it's very nearly ready to be used for real. Here are a few notes.

First, I found that traversing a tree of models can be really fast, and there's little reason not to simply traverse the tree. E.g., keeping an array of models and acting on each element of the array is likely more expensive that crawling through the whole tree every time the sim engine needs to do something. Julia's happy when things are on the stack. This code, therefore, is all about the tree.

Second, writing model-specific methods for functions defined in the simulation engine is lame. It makes for ugly, not-terribly-portable code. The user should be able to model stuff _entirely_ without using types from the simulation engine. This imposes the least restrictions on how code is written, and it maximizes the potential for code to be useful outside of the simulation, such as when simply analyzing (not simulating) models.

Third, immutables are great, especially in the context of simulation. It would be great if a user _couldn't_ update their model directly. It would be great if the function you called was `simulate` and not `simulate!`. Also, NamedTuples have gotten _really_ fast and convenient.

Fourth, users want to describe models, describe continuous-time outputs, describe updates, etc., to the simulation engine. These descriptions are outputs from user functions. The simulation engine is responsible for using the descriptions to make the right thing happen. It can then build a new immutable "model" -- constants and state all baked in -- that it passes to the user's functions. Starting from initialization, which describes the initial model, this cycle continues until the end time.

Fifth, the simulation engine should provide a very small set of features that can be used for a variety of things. Then, it can provide a few convenience functions that simply wrap these features up into familiar patterns. E.g., the simulation doesn't need to know that a model is regularly sampled. However, it can provide a convenience function to the user that just uses lower-level stuff, so that it's easy for the user to make a regularly-sampled model.

`overdot.jl` contains the draft of the Overdot module. However, I'd recommend reading over `overdot-demo-1.jl` first. It motivates what's in `Overdot`. The demo is too brief for general audiences. Right now, it's just for us.
