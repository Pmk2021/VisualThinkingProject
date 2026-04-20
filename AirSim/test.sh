import airsim

client = airsim.VehicleClient()
client.confirmConnection()

image_response = client.simGetImage("0", airsim.ImageType.Scene)
with open("airsim_snapshot.png", "wb") as f:
    f.write(image_response)

print("Got image!")