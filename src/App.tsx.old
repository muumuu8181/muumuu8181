// Previous imports and code remain the same until the mg selection buttons section...
              {selectedItem?.name === item && (
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Scale className="h-6 w-6 flex-shrink-0" />
                    <span className="text-lg">摂取量:</span>
                  </div>
                  <div className="grid grid-cols-3 gap-2">
                    {AMOUNT_OPTIONS.map((amount) => (
                      <Button
                        key={amount}
                        variant={selectedItem.amount === amount ? "default" : "outline"}
                        className="h-14 text-lg w-full"
                        onClick={() => handleAmountSelect(amount)}
                      >
                        {amount}
                      </Button>
                    ))}
                  </div>
                </div>
              )}
// Rest of the file remains the same...
