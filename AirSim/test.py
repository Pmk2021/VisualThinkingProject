import airsim

client = airsim.VehicleClient()
client.confirmConnection()

responses = client.simGetImages([
    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
])

print(len(responses))
print(responses[0].width, responses[0].height)
with open("airsim_snapshot.png", "wb") as f:
    f.write(responses[0].image_data_uint8)