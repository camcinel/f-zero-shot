function get_checkpoint()
    local checkpoint = data.curr_checkpoint
    local lap = data.lap_number
    local lap_size = data.lap_size
    if lap == 255 then
        return 0
    else
        return checkpoint + lap * lap_size
    end
end

data.prev_checkpoint = get_checkpoint()
function checkpoint_reward()
    local new_checkpoint = get_checkpoint()

    local reward = (new_checkpoint - data.prev_checkpoint) * 1
    data.prev_checkpoint = new_checkpoint
    return reward
end

function get_direction()
    if data.direction == 4 then
        return 1
    elseif data.direction == 5 then
        return -1
    else
        return 0
    end
end

function speed_with_direction()
    if data.speed > 10000 then
        return 0
    else
        return get_direction() * (data.speed / 45000)
    end
end

prev_health = 239 -- starting health
function health_change()
    if data.health < prev_health then
        local delta = prev_health - data.health
        prev_health = data.health
        return delta
    else
        return 0
    end
end

function total_reward()
    return speed_with_direction() + checkpoint_reward() - 0.1 * health_change()
end