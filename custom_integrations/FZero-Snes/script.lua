function speed_with_direction()
    if data.speed > 10000 then
        return 0
    elseif data.direction == 4 then
        return data.speed / 1000.0
    elseif data.direction == 5 then
        return -1.0 * data.speed / 1000.0
    else
        return 0
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
    return speed_with_direction() - 10.0 * health_change()
end